import os
import glob
import argparse
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from olvae.util import instantiate_from_config


torch.set_float32_matmul_precision("medium")


def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


class ImageFolderHWC(Dataset):
    """
    Returns dict compatible with LiteAutoencoderKL.get_input:
      {"image": (H,W,C) float32 in [-1,1]}
    """
    def __init__(self, folder: str, size_hw=(256, 256)):
        super().__init__()
        self.paths = list_images(folder)
        if len(self.paths) == 0:
            raise ValueError(f"No images found in: {folder}")
        self.size_hw = tuple(size_hw)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            # PIL resize expects (W,H)
            img = img.resize((self.size_hw[1], self.size_hw[0]), resample=Image.BICUBIC)
            arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,C
            x = torch.from_numpy(arr) * 2.0 - 1.0            # [-1,1]
            return {"image": x}
        except Exception:
            return self.__getitem__((idx + 1) % len(self.paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--ckpt", type=str, default=None)  # file path to .ckpt
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=256)

    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)  # keep 1 unless you explicitly want DDP
    parser.add_argument("--precision", type=str, default="bf16-mixed")  # H100-friendly
    parser.add_argument("--val_every_n_epochs", type=int, default=1)

    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_dir = os.path.join(args.data_root, "training")
    val_dir = os.path.join(args.data_root, "testing")

    # --- model from YAML
    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.model
    model = instantiate_from_config(model_cfg)

    base_lr = float(model_cfg.get("base_learning_rate", 1e-4))
    model.learning_rate = float(args.lr) if args.lr is not None else base_lr

    if args.ckpt is not None:
        model.init_from_ckpt(args.ckpt, ignore_keys=[])

    # --- data
    train_ds = ImageFolderHWC(train_dir, size_hw=(args.img_h, args.img_w))
    val_ds = ImageFolderHWC(val_dir, size_hw=(args.img_h, args.img_w))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # --- callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, "checkpoints"),
        filename="litevae-{epoch:03d}-{step:08d}",
        save_last=True,
        save_top_k=2,
        monitor="val/rec_loss",
        mode="min",
        every_n_epochs=1,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # If you use >1 GPU, enable sync_dist in your model logging for epoch-level metrics.
    # (Or keep devices=1 to avoid the distributed complexity for now.)

    trainer_kwargs = dict(
        default_root_dir=args.out_dir,
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=args.precision,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=50,
        check_val_every_n_epoch=args.val_every_n_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        enable_progress_bar=True,
    )

    if args.max_steps and args.max_steps > 0:
        trainer_kwargs["max_steps"] = args.max_steps

    # strategy: only set DDP if you actually set devices>1
    if args.devices > 1:
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_false"

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
