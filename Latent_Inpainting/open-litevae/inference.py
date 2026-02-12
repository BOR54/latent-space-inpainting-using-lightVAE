import os
import glob
import argparse
from typing import List, Dict

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision.utils import save_image

from olvae.util import instantiate_from_config


def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


def to_01(x: torch.Tensor) -> torch.Tensor:
    # [-1, 1] -> [0, 1]
    return (x.clamp(-1, 1) + 1.0) / 2.0


def load_image_as_hwc(path: str, h: int, w: int) -> torch.Tensor:
    """
    Returns (H, W, C) float32 in [-1, 1]
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256), resample=Image.BICUBIC)  # PIL expects (W,H)
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3) in [0,1]
    x = torch.from_numpy(arr)                         # (H,W,3)
    x = x * 2.0 - 1.0                                 # [-1,1]
    return x


def save_triplet_grid(x_bchw: torch.Tensor, xrec_bchw: torch.Tensor, out_path: str) -> None:
    """
    Saves a grid containing:
      row 1: inputs
      row 2: reconstructions
      row 3: abs error
    """
    err = (x_bchw - xrec_bchw).abs().clamp(0, 1)  # [0,1]
    grid = torch.cat([to_01(x_bchw), to_01(xrec_bchw), err], dim=0)  # (3B,3,H,W)
    save_image(grid, out_path, nrow=x_bchw.shape[0])


def find_latest_ckpt(ckpt_dir: str) -> str:
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No .ckpt files found in: {ckpt_dir}")
    # Lightning filenames often include step; but safe default is newest mtime
    ckpts = sorted(ckpts, key=lambda p: os.path.getmtime(p))
    return ckpts[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/bmutembei36/Latent_Inpainting/open-litevae/configs/olitevaeB_im_f8c12.yaml",
        help="Path to LiteVAE config YAML",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/scratch/bmutembei36/litevae_runs/checkpoints/",
        help="Directory containing Lightning .ckpt files",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Specific checkpoint path. If not set, picks latest in ckpt_dir.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/bmutembei36/Data/Data",
        help="Data root containing training/ and testing/",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["training", "testing"],
        default="testing",
        help="Which folder to run inference on",
    )
    parser.add_argument("--out_dir", type=str, default="/scratch/bmutembei36/litevae_infer_outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=50, help="How many batches to run (cap).")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        help="If set, sample z ~ q(z|x). Otherwise use posterior mean (mode).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- choose checkpoint
    ckpt_path = args.ckpt if args.ckpt is not None else find_latest_ckpt(args.ckpt_dir)

    # ----- load config + instantiate model
    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.model
    model = instantiate_from_config(model_cfg)

    # Lightning checkpoint stores weights under "state_dict"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # load into your module (your override tolerates missing/unexpected keys)
    model.load_state_dict(state, strict=False)

    model = model.to(device)
    model.eval()

    # ----- data
    split_dir = os.path.join(args.data_root, args.split)
    paths = list_images(split_dir)
    if len(paths) == 0:
        raise ValueError(f"No images found in: {split_dir}")

    print("Using device:", device)
    print("Checkpoint:", ckpt_path)
    print(f"Found {len(paths)} images in {split_dir}")
    print("Saving outputs to:", args.out_dir)

    # ----- inference loop
    with torch.no_grad():
        bi = 0
        idx = 0
        while bi < args.num_batches and idx < len(paths):
            batch_paths = paths[idx: idx + args.batch_size]
            idx += args.batch_size
            if len(batch_paths) == 0:
                break

            # build batch in HWC then convert to BCHW via model.get_input style
            x_list = [load_image_as_hwc(p, args.height, args.width) for p in batch_paths]
            x_hwc = torch.stack(x_list, dim=0)  # (B,H,W,C)
            x = x_hwc.permute(0, 3, 1, 2).contiguous().float().to(device)  # (B,3,H,W)

            # forward
            xrec, posterior = model(x, sample_posterior=args.sample_posterior)
            z = posterior.sample() if args.sample_posterior else posterior.mode()

            out_path = os.path.join(args.out_dir, f"{args.split}_batch_{bi:04d}.png")
            save_triplet_grid(x, xrec, out_path)

            print(
                f"[{bi:04d}] x={tuple(x.shape)} z={tuple(z.shape)} xrec={tuple(xrec.shape)} -> {out_path}"
            )

            bi += 1

    print("Done.")


if __name__ == "__main__":
    main()
