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


#Utilizing tensor cores on gpu to improve performance at the cost of precision
torch.set_float32_matmul_precision('medium')

def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


def pad_to_multiple(x_hwc: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    """
    Pads H and W to be divisible by `multiple` using reflect padding.
    Input: (H, W, C) float tensor.
    Output: (H_pad, W_pad, C)
    """
    h, w, c = x_hwc.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    if pad_h == 0 and pad_w == 0:
        return x_hwc

    # torch.nn.functional.pad expects (..., H, W) for 2D, so permute to CHW temporarily
    x_chw = x_hwc.permute(2, 0, 1)  # C,H,W
    # Pad order is (left, right, top, bottom) for last two dims in F.pad with 2D
    # We'll pad only on bottom and right to keep "current" content intact.
    x_chw = torch.nn.functional.pad(x_chw, (0, pad_w, 0, pad_h), mode="reflect")
    return x_chw.permute(1, 2, 0)  # back to H,W,C


class ImageFolderHWC(Dataset):
    """
    Returns dict batches compatible with LiteAutoencoderKL.get_input:
    {"image": (H,W,C) float32 in [-1,1]}
    Keeps original image dimensions (no resize). Optionally pads to multiple-of-8.
    """
    def __init__(self, folder: str, pad_mult: int = 8):
        super().__init__()
        self.paths = list_images(folder)
        if len(self.paths) == 0:
            raise ValueError(f"No images found in: {folder}")
        self.pad_mult = pad_mult

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            
            # force every image to (H,W) = (256, 256)
            img = img.resize((256, 256), resample=Image.BICUBIC)
            
            arr = np.asarray(img).astype(np.float32) / 255.0
            x = torch.from_numpy(arr)
            x = x * 2.0 - 1.0

            if self.pad_mult is not None and self.pad_mult > 1:
                x = pad_to_multiple(x, multiple=self.pad_mult)

            return {"image": x}

        except Exception as e:
#            print(f"Error loading {p}: {e}")
            # Recursively try the next index to skip the corrupted file
            return self.__getitem__((idx + 1) % len(self.paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/bmutembei36/Latent_Inpainting/open-litevae/configs/olitevaeB_im_f8c12.yaml",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/bmutembei36/Data/Data",
        help="Data root containing training/ and testing/ folders",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1, help="If >0, caps total steps (overrides epochs).")
    parser.add_argument("--lr", type=float, default=None, help="Override base_learning_rate from YAML if set.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint to resume/init from.")
    parser.add_argument("--out_dir", type=str, default="litevae_train_runs")
    parser.add_argument("--val_every_n_epochs", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="1.0=all, or 0.1 for 10%")
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "training")
    val_dir = os.path.join(args.data_root, "testing")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- config + model
    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.model

    model = instantiate_from_config(model_cfg)

    # Lightning expects `learning_rate` attribute (your model uses self.learning_rate in configure_optimizers)
    base_lr = float(model_cfg.get("base_learning_rate", 1e-4))
    model.learning_rate = float(args.lr) if args.lr is not None else base_lr

    if args.ckpt is not None:
        model.init_from_ckpt(args.ckpt, ignore_keys=[])

    # --- data
    train_ds = ImageFolderHWC(train_dir, pad_mult=8)
    val_ds = ImageFolderHWC(val_dir, pad_mult=8)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # --- callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, "checkpoints"),
        filename="litevae-{epoch:03d}-{step:08d}",
        save_top_k=2,
        monitor="val/rec_loss",  # can set later to "val/rec_loss" after first run
        every_n_epochs=1,
        save_last=True,
    )

    lr_cb = LearningRateMonitor(logging_interval="step")

    # --- trainer
    trainer_kwargs = dict(
        default_root_dir=args.out_dir,
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,               # Change from 1 to 2 to use both H100s
        strategy="auto",          # Use Distributed Data Parallel for multi-GPU
        callbacks=[ckpt_cb, lr_cb],
        precision="bf16-mixed",  # H100s excel at bf16 precision
        # Disable the standard progress bar for cleaner logs on clusters
        enable_progress_bar=True,
        log_every_n_steps=100,
        check_val_every_n_epoch=args.val_every_n_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        
    )

    if args.max_steps and args.max_steps > 0:
        trainer_kwargs["max_steps"] = args.max_steps

    trainer = pl.Trainer(**trainer_kwargs)

    # --- fit
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
