import argparse
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

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
    args.test_csv_path = str(dataset_root / "prompt" / "test.csv")
    args.test_root_path = str(dataset_root / "Test")
    return args


if __name__ == "__main__":
    args = get_parser()
    set_seed(args.seed)

    model = TextmoeSegWrapper(args)
    checkpoint = torch.load(
        "./semi_supervised/semi_supervised.ckpt", map_location="cpu"
    )["state_dict"]
    model.load_state_dict(checkpoint, strict=True)

    ds_test = QaTa(
        csv_path=args.test_csv_path,
        root_path=args.test_root_path,
        tokenizer=args.bert_type,
        image_size=args.image_size,
        mode="test",
        dataset_id=0,
        seed=args.seed,
    )
    dl_test = DataLoader(
        ds_test, batch_size=args.valid_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init, generator=torch.Generator().manual_seed(args.seed),
    )

    trainer = pl.Trainer(accelerator="gpu", devices=args.device)
    model.eval()
    trainer.test(model, dl_test)
