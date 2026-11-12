import argparse
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from engine.wrapper import TextmoeSegWrapper
from utils.dataset import QaTa, worker_init, set_seed
import utils.config as config


ROOT = Path(__file__).resolve().parent


def get_parser():
    parser = argparse.ArgumentParser(description="Textmoe")
    parser.add_argument("--config", default=str(ROOT / "config" / "training.yaml"))
    arguments = parser.parse_args()
    args = config.load_cfg_from_cfg_file(arguments.config)

    dataset_name = "QaTa-COV19-v2" if args.dataset == "qata" else "MosMed"
    dataset_root = Path(args.data_root) / dataset_name
    args.train_csv_path = str(dataset_root / "prompt" / "train.csv")
    args.train_root_path = str(dataset_root / "Train")
    return args


if __name__ == "__main__":
    args = get_parser()
    set_seed(args.seed)
    print("cuda:", torch.cuda.is_available())

    print("Preparing labeled data...")
    ds_labeled = QaTa(
        csv_path=args.train_csv_path,
        root_path=args.train_root_path,
        tokenizer=args.bert_type,
        image_size=args.image_size,
        mode="pretrain",
        dataset_id=0,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
    )
    ds_valid = QaTa(
        csv_path=args.train_csv_path,
        root_path=args.train_root_path,
        tokenizer=args.bert_type,
        image_size=args.image_size,
        mode="valid",
        dataset_id=0,
        seed=args.seed,
    )
    dl_labeled = DataLoader(
        ds_labeled, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init, generator=torch.Generator().manual_seed(args.seed),
    )
    dl_valid = DataLoader(
        ds_valid, batch_size=args.valid_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init, generator=torch.Generator().manual_seed(args.seed),
    )

    print("Start pre-training with labeled data...")
    model = TextmoeSegWrapper(args)
    model_ckpt_pre = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    early_stopping_pre = EarlyStopping(
        monitor="val_loss", patience=args.patience, mode="min"
    )
    trainer_pre = pl.Trainer(
        logger=True,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.device,
        callbacks=[model_ckpt_pre, early_stopping_pre],
        enable_progress_bar=False,
    )
    trainer_pre.fit(model, dl_labeled, dl_valid)
    print("Pre-training complete.")

    print("Preparing for semi-supervised training...")
    model = TextmoeSegWrapper(args)
    checkpoint = torch.load(model_ckpt_pre.best_model_path, map_location="cpu")["state_dict"]
    model.load_state_dict(checkpoint, strict=True)

    ds_semi = QaTa(
        csv_path=args.train_csv_path,
        root_path=args.train_root_path,
        tokenizer=args.bert_type,
        image_size=args.image_size,
        mode="semi",
        dataset_id=0,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
    )
    dl_semi = DataLoader(
        ds_semi, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init, generator=torch.Generator().manual_seed(args.seed),
    )

    model_ckpt_semi = ModelCheckpoint(
        dirpath="./semi_supervised",
        filename="semi_supervised",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    early_stopping_semi = EarlyStopping(
        monitor="val_loss", patience=args.patience, mode="min"
    )
    trainer_semi = pl.Trainer(
        logger=True,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.device,
        callbacks=[model_ckpt_semi, early_stopping_semi],
        enable_progress_bar=False,
    )
    print("Start semi-supervised training...")
    trainer_semi.fit(model, dl_semi, dl_valid)
    print("Semi-supervised training complete.")
