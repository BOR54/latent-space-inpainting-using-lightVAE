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

# Utilizing tensor cores on GPU
torch.set_float32_matmul_precision("medium")

def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)

def pad_to_multiple(x_hwc: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    h, w, c = x_hwc.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    if pad_h == 0 and pad_w == 0:
        return x_hwc

    x_chw = x_hwc.permute(2, 0, 1)  # C,H,W
    # Use reflect padding for image boundaries to avoid artifacts in wavelet loss
    x_chw = torch.nn.functional.pad(x_chw, (0, pad_w, 0, pad_h), mode="reflect")
    return x_chw.permute(1, 2, 0)  # H,W,C

class ImageFolderHWC(Dataset):
    def __init__(self, folder: str, pad_mult: int = 8, size: int = 256):
        super().__init__()
        self.paths = list_images(folder)
        if len(self.paths) == 0:
            raise ValueError(f"No images found in: {folder}")
        self.pad_mult = pad_mult
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((self.size, self.size), resample=Image.BICUBIC)

            arr = np.asarray(img).astype(np.float32) / 255.0
            x = torch.from_numpy(arr)
            x = x * 2.0 - 1.0  # Scale to [-1, 1]

            if self.pad_mult is not None and self.pad_mult > 1:
                x = pad_to_multiple(x, multiple=self.pad_mult)

            return {"image": x}
        except Exception as e:
            print(f"Error loading {p}: {e}")
            return self.__getitem__((idx + 1) % len(self.paths))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="litevae_train_runs")
    parser.add_argument("--val_every_n_epochs", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "training")
    val_dir = os.path.join(args.data_root, "testing")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Config & Model Setup ---
    cfg = OmegaConf.load(args.config)
    model = instantiate_from_config(cfg.model)

    # Manual optimization is handled inside the model's training_step
    model.automatic_optimization = False

    # Set learning rate
    if args.lr is not None:
        model.learning_rate = args.lr
    else:
        # Default fallback if not in config
        model.learning_rate = getattr(cfg.model, "base_learning_rate", 4.5e-6)

    # --- DataLoaders ---
    train_ds = ImageFolderHWC(train_dir, pad_mult=8)
    val_ds = ImageFolderHWC(val_dir, pad_mult=8)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # --- Callbacks ---
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="litevae-{epoch:02d}-{step:06d}",
        save_top_k=3,
        monitor="val/rec_loss",
        mode="min",
        save_last=True,
    )

    lr_cb = LearningRateMonitor(logging_interval="step")

    # --- Trainer Setup ---
    trainer = pl.Trainer(
        default_root_dir=args.out_dir,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[ckpt_cb, lr_cb],
        precision=args.precision,
        log_every_n_steps=50,
        check_val_every_n_epoch=args.val_every_n_epochs,
        num_sanity_val_steps=1, # Important to catch GAN shape mismatches early
    )

    # --- Fit ---
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)

if __name__ == "__main__":
    main()