from copy import deepcopy
import datetime
import sys
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from monai.losses import DiceLoss
from utils.dataset import Metrics
from utils.model import TextmoeSeg

class BinaryDiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)

    def forward(self, logits, target):
        return self.dice(logits, target) + F.binary_cross_entropy_with_logits(logits, target)

class SemiStrategy:
    def prepare(self, image, text):
        batch_size = image.size(0)
        contrast = torch.empty((batch_size, 1, 1, 1), device=image.device).uniform_(0.8, 1.2)
        student_image = image * contrast
        cutout_size = int(student_image.size(-1) * 0.2)
        if cutout_size > 0:
            for index in range(batch_size):
                # 保留历史版本的随机数顺序，避免改变后续增强结果。
                torch.rand((), device=image.device)
                cutout_y = torch.randint(0, student_image.size(2) - cutout_size, (1,)).item()
                cutout_x = torch.randint(0, student_image.size(3) - cutout_size, (1,)).item()
                student_image[index, :, cutout_y:cutout_y + cutout_size, cutout_x:cutout_x + cutout_size] = 0
        student_text = {}
        token_mask = None
        for key, value in text.items():
            if key == 'input_ids' and value.dtype in (torch.int, torch.long):
                token_mask = (torch.rand_like(value, dtype=torch.float) < 0.15) & text['attention_mask'].bool()
                token_mask[:, 0] = False
                token_mask.scatter_(1, text['attention_mask'].sum(1, keepdim=True).sub(1).long(), False)
                student_text[key] = value.masked_fill(token_mask, 0)
            elif key == 'attention_mask' and token_mask is not None:
                student_text[key] = value.masked_fill(token_mask, 0)
            else:
                student_text[key] = value.clone()
        return student_image, student_text

    def rampup(self, epoch):
        progress = min(epoch / 10.0, 1.0)
        return float(torch.exp(torch.tensor(-5.0 * (1.0 - progress) ** 2)))

class TextmoeSegWrapper(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.model = TextmoeSeg(args.bert_type, args.vision_type, args.project_dim)
        self.ema_model = deepcopy(self.model).requires_grad_(False).eval()
        self.loss_fn = BinaryDiceBCELoss()
        self.semi_strategy = SemiStrategy()
        self.rcea_expert_sep_margin = 0.2
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.max_epochs = args.max_epochs
        self.ema_step = 0
        self.history = {}
        self.val_metrics = Metrics()
        self.test_metrics = Metrics()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=0.000001,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.model(x)

    def train(self, mode=True):
        super().train(mode)
        self.ema_model.eval()
        return self

    @torch.no_grad()
    def update_ema(self, step):
        self.update_ema_variables(self.model, self.ema_model, step)
        buffers = dict(self.model.named_buffers())
        for name, buffer in self.ema_model.named_buffers():
            buffer.copy_(buffers[name])
        self.ema_model.eval()

    def shared_step(self, batch, batch_idx):
        if self.trainer.training:
            x, target, labeled = batch
            image, text, dataset_id = x
            target = target.float()
            labeled = labeled > 0
            total = image.new_zeros(())
            if labeled.any():
                labeled_text = {}
                for key, value in text.items():
                    labeled_text[key] = value[labeled]
                logits, aux = self.model([
                    image[labeled],
                    labeled_text,
                    dataset_id[labeled],
                ])
                y = target[labeled]
                seg = self.loss_fn(logits, y)
                total = seg + 0.01 * self.gate_balance_loss(aux)
            unlabeled = ~labeled
            if unlabeled.any():
                x = image[unlabeled]
                txt = {}
                for key, value in text.items():
                    txt[key] = value[unlabeled]
                ids = dataset_id[unlabeled]
                with torch.no_grad():
                    tp = self.ema_model([x, txt, ids])[0].sigmoid()
                    pseudo_target = (tp > 0.5).float()
                student_image, student_text = self.semi_strategy.prepare(x, txt)
                logits, aux = self.model([student_image, student_text, ids])
                unsup = self.loss_fn(logits, pseudo_target) + 0.01 * self.gate_balance_loss(aux)
                total = total + self.semi_strategy.rampup(self.current_epoch) * unsup
            loss = total
            preds, y = None, None
        else:
            x, y, flag = batch
            logits, auxiliary = self(x)
            loss = self.loss_fn(logits, y.float())
            preds, y = logits.detach().sigmoid(), y.detach()

        return {"loss": loss, "preds": preds, "y": y, "batch_size": len(batch[1])}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and isinstance(batch[0], (list, tuple)):
            return self(batch[0])
        return self(batch)

    def shared_step_end(self, outputs, stage):
        # B0 只统计验证/测试指标，不把无标签占位掩码当成真实标签。
        if stage != "train":
            metrics = self.val_metrics if stage == "val" else self.test_metrics
            metrics.update(outputs["preds"], outputs["y"])
        return outputs["loss"].mean()

    def training_step_end(self, outputs):
        loss = self.shared_step_end(outputs, "train")
        self.log("train_loss_step", loss, on_step=True, on_epoch=False,
                 batch_size=outputs["batch_size"])
        return {"loss": loss, "batch_size": outputs["batch_size"]}

    def validation_step_end(self, outputs):
        return {"val_loss": self.shared_step_end(outputs, "val"),
                "batch_size": outputs["batch_size"]}

    def test_step_end(self, outputs):
        return {"test_loss": self.shared_step_end(outputs, "test"),
                "batch_size": outputs["batch_size"]}

    def shared_epoch_end(self, outputs, stage="train"):
        epoch = self.trainer.current_epoch
        key = (stage + "_loss").replace("train_", "")
        # 保持 B0 按样本数加权的 epoch loss，避免末批大小改变模型选择。
        total_loss = torch.zeros((), device=self.device)
        count = 0
        for output in outputs:
            total_loss += output[key].detach() * output["batch_size"]
            count += output["batch_size"]
        dic = {"epoch": epoch, stage + "_loss": (total_loss / count).item()}

        if stage != "train":
            metrics = self.val_metrics if stage == "val" else self.test_metrics
            for name, value in metrics.compute().items():
                dic[stage + "_" + name] = value
            if stage == "val":
                self.val_metrics = Metrics()
            else:
                self.test_metrics = Metrics()
        if stage != "test" and not self.trainer.sanity_checking:
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic

    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="train")
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)
        self.log("train_loss_epoch", dic["train_loss"])

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

        ckpt_cb = self.trainer.checkpoint_callback
        if ckpt_cb is None or self.trainer.sanity_checking:
            return
        monitor = ckpt_cb.monitor
        mode = ckpt_cb.mode
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor, arr_scores[best_score_idx]), file=sys.stderr)

    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="test")
        dic.pop("epoch", None)
        self.print(dic)
        self.log_dict(dic, logger=True)

    def get_history(self):
        return pd.DataFrame(self.history.values())

    def print_bar(self):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.print("\n" + "=" * 80 + nowtime)

    def on_train_batch_end(self, outputs, batch, batch_index):
        self.update_ema(self.ema_step)
        self.ema_step += 1

    @staticmethod
    def update_ema_variables(model, ema_model, global_step, alpha=0.999):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.data.mul_(alpha).add_(parameter.data, alpha=1 - alpha)

    @staticmethod
    def gate_balance_loss(auxiliary):
        losses = []
        for weights in auxiliary['gate_weights']:
            if weights.ndim not in (2, 4):
                raise ValueError(f'Invalid routing weights shape: {tuple(weights.shape)}')
            reduction_dims = (0,) + tuple(range(2, weights.ndim))
            target = torch.full((weights.size(1),), 1.0 / weights.size(1), device=weights.device, dtype=weights.dtype)
            losses.append(F.mse_loss(weights.mean(dim=reduction_dims), target))
        return torch.stack(losses).mean()

    @staticmethod
    def _single_channel(tensor, binary=False):
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        tensor = tensor.float().mean(dim=1, keepdim=True)
        return (tensor > 0.5).float() if binary else tensor

    def rcea_supervised_loss(self, auxiliary, target):
        target = self._single_channel(target, binary=True)
        losses = []
        for item in auxiliary['rcea_aux']:
            score = F.interpolate(item['rcea_score'], size=target.shape[-2:], mode='bilinear', align_corners=False)
            losses.append(F.binary_cross_entropy_with_logits(score, target))
        return torch.stack(losses).mean()

    @staticmethod
    def _masked_text_mean(tokens, mask):
        weight = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weight).sum(dim=-2) / weight.sum(dim=-2).clamp_min(1.0)

    def rcea_expert_specialization_loss(self, auxiliary, target):
        separation = []
        lesion = self._single_channel(target, binary=True)
        for item in auxiliary['rcea_aux'][1:4]:
            outputs = item['rcea_expert_outputs']
            batch, experts, _, height, width = outputs.shape
            mask = F.interpolate(lesion, size=(height, width), mode='nearest').to(outputs.dtype)
            foreground = mask.flatten(1).sum(dim=-1)
            valid = foreground > 0
            if not valid.any():
                continue
            weight = mask[:, None]
            denominator = foreground[:, None, None].clamp_min(1.0)
            corrected = (outputs * weight).flatten(3).sum(dim=-1) / denominator
            positive = self._masked_text_mean(item['rcea_positive_text'], item['rcea_positive_mask'])
            negatives = self._masked_text_mean(item['rcea_negative_text'], item['rcea_negative_mask'])
            factor_ids = item['rcea_expert_negative_factors']
            corrected = F.normalize(corrected.float(), dim=-1)
            positive = F.normalize(positive.float(), dim=-1)
            negatives = F.normalize(negatives.float(), dim=-1)
            assigned = negatives.index_select(1, factor_ids)
            positive_similarity = torch.einsum('bec,bc->be', corrected, positive)
            negative_similarity = (corrected * assigned).sum(dim=-1)
            separation.append(F.relu(self.rcea_expert_sep_margin + negative_similarity[valid] - positive_similarity[valid]).mean())
        zero = target.sum() * 0.0
        if not separation:
            return zero
        return torch.stack(separation).mean()

    def rcea_cross_negative_loss(self, auxiliary):
        losses = []
        for item in auxiliary['rcea_aux']:
            visual = item['rcea_expert_conv_features']
            batch, experts, _, _, _ = visual.shape
            if experts != 3:
                raise ValueError(f'cross-negative loss requires 3 experts, got {experts}')
            visual = F.normalize(visual.float().mean(dim=(-2, -1)), dim=-1)
            negatives = self._masked_text_mean(item['rcea_negative_text'], item['rcea_negative_mask'])
            negatives = F.normalize(negatives.float().detach(), dim=-1)
            negative_similarity = torch.einsum('bec,bfc->bef', visual, negatives)
            cross_expert = ~torch.eye(experts, dtype=torch.bool, device=visual.device)
            losses.append(negative_similarity.masked_select(cross_expert.unsqueeze(0).expand(batch, -1, -1)).mean())
        return torch.stack(losses).mean()
