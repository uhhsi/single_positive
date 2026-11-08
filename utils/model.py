import math
import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoModel

class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim=768):
        super().__init__()
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True, local_files_only=True)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        text_output = output['hidden_states'][-1]
        return text_output

class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim=768):
        super().__init__()
        self.model = AutoModel.from_pretrained(vision_type, output_hidden_states=True, local_files_only=True)

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        return {'feature': output['hidden_states']}

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=5000):
        super().__init__()
        encoding = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer('pe', encoding.unsqueeze(0))

    def forward(self, tokens):
        return tokens + self.pe[:, :tokens.size(1)]

class ReliabilityCounterfactualExpertAttention(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f'RCEA dim={dim} must divide heads={num_heads}')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.query_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        gate_dim = 2 * dim + 1
        self.gate = nn.Sequential(nn.LayerNorm(gate_dim), nn.Linear(gate_dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _split_heads(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _valid_mask(mask, batch, length, device):
        if mask is None:
            return torch.ones(batch, length, dtype=torch.bool, device=device)
        if tuple(mask.shape) != (batch, length):
            raise ValueError(f'text mask {tuple(mask.shape)} != {(batch, length)}')
        valid = mask.to(device=device).bool()
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        return valid

    def _attend(self, query, text, mask):
        batch, length, _ = text.shape
        normalized = self.text_norm(text)
        key = self._split_heads(self.key_proj(normalized))
        value = self._split_heads(self.value_proj(normalized))
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        valid = self._valid_mask(mask, batch, length, text.device)
        logits = logits.masked_fill(~valid[:, None, None, :], torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        context = torch.matmul(attention, value)
        valid_count = valid.sum(dim=-1).clamp_min(1).to(logits.dtype)
        evidence = torch.logsumexp(logits, dim=-1) - valid_count.log()[:, None, None]
        return (context, evidence)

    def forward(self, visual_tokens, positive_text, negative_text, positive_mask=None, negative_mask=None):
        query = self._split_heads(self.query_proj(self.query_norm(visual_tokens)))
        positive_context, positive_evidence = self._attend(query, positive_text, positive_mask)
        negative_context, negative_evidence = self._attend(query, negative_text, negative_mask)
        batch = visual_tokens.size(0)
        spatial_length = visual_tokens.size(1)
        context = (positive_context - negative_context).transpose(1, 2)
        context = context.reshape(batch, spatial_length, self.dim)
        context = self.output_proj(context)
        score = (positive_evidence - negative_evidence).mean(dim=1).unsqueeze(-1)
        gate_input = torch.cat([visual_tokens, context, score], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        output = visual_tokens + self.residual_scale * gate * context
        return (output, {'rcea_score_tokens': score, 'rcea_gate_tokens': gate})

class ContextExpertConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(out_channels, out_channels, kernel_size=1))

    def forward(self, image):
        return self.block(image)

class BoundaryExpertConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.edge_project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.edge_scale = nn.Parameter(torch.tensor(0.5))
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, image):
        channels = image.size(1)
        weight_x = self.sobel_x.to(image.dtype).repeat(channels, 1, 1, 1)
        weight_y = self.sobel_y.to(image.dtype).repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(image, weight_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, weight_y, padding=1, groups=channels)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-06)
        return self.base(image) + self.edge_scale * self.edge_project(edge)

class MoeConv(nn.Module):

    def __init__(self, in_channels, out_channels, text_length=24):
        super().__init__()
        self.text_project = nn.Sequential(nn.Linear(768, out_channels), nn.GELU(), nn.LayerNorm(out_channels), nn.LeakyReLU(0.1))
        self.txt_pos = PositionalEncoding(out_channels, max_len=text_length)
        self.image_gate_project = nn.Linear(in_channels, out_channels)
        gate_dim = in_channels + out_channels + 1
        self.gate = nn.Sequential(nn.LayerNorm(gate_dim), nn.Linear(gate_dim, max(gate_dim // 2, 3)), nn.GELU(), nn.Linear(max(gate_dim // 2, 3), 3))
        convolutions = [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
        convolutions += [ContextExpertConv(in_channels, out_channels), BoundaryExpertConv(in_channels, out_channels)]
        self.experts = nn.ModuleList()
        for convolution in convolutions:
            modules = dict(conv=convolution, vis_pos=PositionalEncoding(out_channels))
            modules['rcea'] = ReliabilityCounterfactualExpertAttention(out_channels, num_heads=2)
            self.experts.append(nn.ModuleDict(modules))
        # 旧 checkpoint 含此参数；保留以支持 strict=True 加载，不参与前向计算。
        self.neg_text_embed = nn.Parameter(torch.zeros(1, out_channels))

    @staticmethod
    def _masked_mean(tokens, mask):
        weight = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-06)

    def forward(self, image, positive_text, negative_text, positive_mask, negative_mask):
        batch, _, height, width = image.shape
        pooled = image.mean(dim=(2, 3))
        positive_projected = self.text_project(positive_text)
        text_pooled = self._masked_mean(positive_projected, positive_mask)
        positive_tokens = self.txt_pos(positive_projected)
        _, factors, length, _ = negative_text.shape
        negative_projected = self.text_project(negative_text)
        channels = negative_projected.size(-1)
        negative_tokens = self.txt_pos(negative_projected.reshape(batch * factors, length, channels)).reshape(batch, factors, length, channels)
        image_semantic = self.image_gate_project(pooled)
        similarity = F.cosine_similarity(image_semantic, text_pooled, dim=-1).unsqueeze(-1)
        weights = F.softmax(self.gate(torch.cat([pooled, text_pooled, similarity], dim=-1)), dim=-1)
        outputs, features, scores, gates = ([], [], [], [])
        for index, expert in enumerate(self.experts):
            visual = expert['conv'](image)
            features.append(visual)
            tokens = expert['vis_pos'](visual.permute(0, 2, 3, 1).reshape(batch, height * width, visual.size(1)))
            fused, aux = expert['rcea'](tokens, positive_tokens, negative_tokens[:, index], positive_mask, negative_mask[:, index])
            scores.append(aux['rcea_score_tokens'].reshape(batch, height, width, 1).permute(0, 3, 1, 2))
            gates.append(aux['rcea_gate_tokens'].reshape(batch, height, width, 1).permute(0, 3, 1, 2))
            outputs.append(fused.reshape(batch, height, width, -1).permute(0, 3, 1, 2))
        stacked = torch.stack(outputs, dim=1)
        mixture = weights[:, :, None, None, None]
        output = (mixture * stacked).sum(dim=1)
        aux = dict(rcea_score=(mixture * torch.stack(scores, dim=1)).sum(dim=1), rcea_gate=(mixture * torch.stack(gates, dim=1)).sum(dim=1), rcea_expert_conv_features=torch.stack(features, dim=1), rcea_expert_outputs=stacked, rcea_positive_text=positive_tokens, rcea_negative_text=negative_tokens, rcea_positive_mask=positive_mask, rcea_negative_mask=negative_mask, rcea_expert_negative_factors=torch.arange(3, device=image.device))
        return (output, weights, aux)

class UNetDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.moe_conv1 = MoeConv(out_channels * 2, out_channels)
        self.moe_conv2 = MoeConv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip, *text):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x, g1, a1 = self.moe_conv1(x, *text)
        x = F.relu(self.bn1(x))
        x, g2, a2 = self.moe_conv2(x, *text)
        x = F.relu(self.bn2(x))
        return x, g1, g2, a1, a2

class TextmoeSeg(nn.Module):

    def __init__(self, bert_type='microsoft/BiomedVLP-CXR-BERT-specialized', vision_type='facebook/convnext-tiny-224', project_dim=768):
        super().__init__()
        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)
        feature_dim = [768, 384, 192, 96]
        self.block1 = UNetDecoderBlock(feature_dim[0], feature_dim[1])
        self.block2 = UNetDecoderBlock(feature_dim[1], feature_dim[2])
        self.block3 = UNetDecoderBlock(feature_dim[2], feature_dim[3])
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=3)

    def forward(self, data):
        image, text, dataset_id = data
        image_output = self.encoder(image)
        image_features = image_output['feature']
        if len(image_features[0].shape) == 4:
            image_features = image_features[1:]
        pm = text['attention_mask']
        pos = self.text_encoder(text['input_ids'], pm)
        ids, nm = (text['neg_input_ids'], text['neg_attention_mask'])
        batch, factors, length = ids.shape
        neg = self.text_encoder(ids.reshape(batch * factors, length), nm.reshape(batch * factors, length)).reshape(batch, factors, length, -1)
        encoded = (pos, neg, pm, nm)
        x, g1, g2, a1, a2 = self.block1(image_features[-1], image_features[-2], *encoded)
        x, g3, g4, a3, a4 = self.block2(x, image_features[-3], *encoded)
        x1, g5, g6, a5, a6 = self.block3(x, image_features[-4], *encoded)
        x = self.decoder1(x1)
        out = self.out(x)
        return out, dict(gate_weights=(g1, g2, g3, g4, g5, g6), rcea_aux=(a1, a2, a3, a4, a5, a6))
