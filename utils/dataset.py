import os
import random

import numpy as np
import pandas as pd
import torch
from monai.data import set_track_meta
from monai.data.utils import worker_init_fn
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, RandRotated, Resized
from monai.utils import set_determinism
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def set_seed(seed):
    set_determinism(seed=seed)
    set_track_meta(False)


class QaTa(Dataset):
    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode="train",
                 image_size=(224, 224), dataset_id=0, labeled_ratio=25, seed=0):
        super(QaTa, self).__init__()
        self.mode = mode
        self.root_path = root_path
        self.image_size = image_size
        self.dataset_id = dataset_id

        data = pd.read_csv(csv_path)

        # 根据配置比例选择有标签训练图片。相同 seed 会得到相同结果。
        data["labeled"] = 1
        train_rows = data.index[data["split"] == "train"].to_numpy()
        data.loc[train_rows, "labeled"] = 0
        order = np.random.RandomState(seed).permutation(train_rows)
        count = len(train_rows) * labeled_ratio // 100
        data.loc[order[:count], "labeled"] = 1

        if mode == "pretrain":
            data = data[(data["split"] == "train") & (data["labeled"] == 1)]
        elif mode == "semi":
            data = data[data["split"] == "train"]
        elif mode == "valid":
            data = data[data["split"] == "val"]
        else:
            data = data[data["split"] == "test"]
        self.rows = data.to_dict("records")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=True, local_files_only=True
        )
        self.transforms = {}
        for labeled in [True, False]:
            self.transforms[labeled] = self.transform(self.image_size, labeled)

    def transform(self, image_size=(224, 224), labeled=True):
        train = self.mode == "pretrain" or self.mode == "semi"
        keys = ["image", "gt"] if labeled else ["image"]
        transforms = [
            LoadImaged(keys, reader="PILReader"),
            EnsureChannelFirstd(keys),
        ]
        if train:
            modes = ["bicubic", "nearest"] if labeled else ["bicubic"]
            transforms.append(
                RandRotated(keys, range_x=(-np.pi / 9, np.pi / 9), prob=1.0, mode=modes)
            )
        transforms.append(Resized(["image"], image_size, mode="bicubic"))
        if labeled:
            transforms.append(Resized(["gt"], image_size, mode="nearest"))
        transforms.append(NormalizeIntensityd(["image"], channel_wise=True))
        return Compose(transforms)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        labeled = self.mode != "semi" or bool(row["labeled"])
        image_name = row["Image"]
        if image_name.startswith("mask_"):
            image_name = image_name[5:]

        image_path = os.path.join(self.root_path, "Images", image_name)
        mask_path = os.path.join(self.root_path, "GTs", row["Image"])

        data = {"image": image_path}
        if labeled:
            data["gt"] = mask_path
        data = self.transforms[labeled](data)

        image = data["image"].float()
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if labeled:
            gt = (data["gt"][:1] == 255).long()
            gt = gt.repeat(3, 1, 1)
        else:
            gt = torch.zeros(3, self.image_size[0], self.image_size[1], dtype=torch.long)

        caption = row["Description"]
        token_output = self.tokenizer.encode_plus(
            caption, padding="max_length", max_length=24, truncation=True,
            return_attention_mask=True, return_tensors="pt",
        )
        text = {
            "input_ids": token_output["input_ids"].squeeze(0),
            "attention_mask": token_output["attention_mask"].squeeze(0),
        }

        x = [image, text, self.dataset_id]
        flag = int(labeled)
        return x, gt, flag


def worker_init(worker):
    worker_init_fn(worker)
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class Metrics:
    def __init__(self):
        self.intersection = 0
        self.prediction = 0
        self.truth = 0
        self.dice = 0
        self.iou = 0
        self.count = 0

    def update(self, probabilities, truth):
        prediction = probabilities > 0.5
        truth = truth.bool()
        dimensions = tuple(range(1, prediction.ndim))
        intersection = (prediction & truth).sum(dimensions).double()
        predicted = prediction.sum(dimensions).double()
        actual = truth.sum(dimensions).double()
        self.intersection += intersection.sum().item()
        self.prediction += predicted.sum().item()
        self.truth += actual.sum().item()
        total = predicted + actual
        union = predicted + actual - intersection
        self.dice += torch.where(total > 0, 2 * intersection / total.clamp_min(1), 1.0).sum().item()
        self.iou += torch.where(union > 0, intersection / union.clamp_min(1), 1.0).sum().item()
        self.count += len(prediction)

    def compute(self):
        total = self.prediction + self.truth
        union = self.prediction + self.truth - self.intersection
        global_dice = 1 if total == 0 else 2 * self.intersection / total
        global_iou = 1 if union == 0 else self.intersection / union
        return {
            "mean_dice": self.dice / self.count,
            "mean_iou": self.iou / self.count,
            "global_dice": global_dice,
            "global_iou": global_iou,
        }
