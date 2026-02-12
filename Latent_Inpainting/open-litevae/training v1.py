import os
import glob
import argparse
from typing import List

import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.utils import save_image

from olvae.util import instantiate_from_config


def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


def make_transform(image_size: int):
    # outputs: tensor in [-1, 1], shape (3, H, W)
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                         # [0,1]
        transforms.Lambda(lambda t: t * 2.0 - 1.0),     # [-1,1]
    ])


def load_batch(paths: List[str], tfm, device: torch.device) -> torch.Tensor:
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")  # enforce 3 channels
        imgs.append(tfm(img))
    x = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)
    return x


def to_01(x: torch.Tensor) -> torch.Tensor:
    # convert [-1,1] -> [0,1] for saving
    return (x.clamp(-1, 1) + 1.0) / 2.0


def save_triplet_grid(x: torch.Tensor, xrec: torch.Tensor, out_path: str) -> None:
    # abs error scaled to look like your internal logger: [0,1] -> [-1,1] -> [0,1] for saving
    err = (x - xrec).abs().clamp(0, 1)
    grid = torch.cat([to_01(x), to_01(xrec), err], dim=0)  # (3B,3,H,W) in [0,1]
    save_image(grid, out_path, nrow=x.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/bmutembei36/Latent_Inpainting/open-litevae/configs/olitevaeB_im_f8c12.yaml",
        help="Path to LiteVAE config YAML",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/bmutembei36/Data/Data/",
        help="Data root containing training/ and testing/",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["training", "testing"],
        default="testing",
        help="Which folder to sample from",
    )
    parser.add_argument("--out_dir", type=str, default="litevae_demo_outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=3, help="How many batches to run")
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        help="If set, sample z ~ q(z|x). Otherwise use posterior mean (mode).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, runs with randomly initialized weights.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- load config
    cfg = OmegaConf.load(args.config)

    # Your YAML structure:
    # model:
    #   target: ...
    #   params: ...
    model_cfg = cfg.model

    # --- instantiate model using repo helper
    model = instantiate_from_config(model_cfg)  # handles target + params recursively

    # Optional checkpoint loading
    if args.ckpt is not None:
        model.init_from_ckpt(args.ckpt, ignore_keys=[])

    model = model.to(device)
    model.eval()

    # --- data
    split_dir = os.path.join(args.data_root, args.split)
    paths = list_images(split_dir)
    if len(paths) == 0:
        raise ValueError(f"No images found in: {split_dir}")

    tfm = make_transform(args.image_size)

    # --- run
    print(f"Device: {device}")
    print(f"Found {len(paths)} images in {split_dir}")
    print(f"Saving outputs to: {os.path.abspath(args.out_dir)}")

    with torch.no_grad():
        for bi in range(args.num_batches):
            start = bi * args.batch_size
            batch_paths = paths[start:start + args.batch_size]
            if len(batch_paths) == 0:
                break

            x = load_batch(batch_paths, tfm, device)

            # forward
            xrec, posterior = model(x, sample_posterior=args.sample_posterior)

            # latent shape sanity check (posterior.mode() is deterministic)
            z = posterior.mode()

            out_path = os.path.join(args.out_dir, f"{args.split}_batch_{bi:03d}.png")
            save_triplet_grid(x, xrec, out_path)

            print(
                f"[{bi:03d}] x={tuple(x.shape)}  z={tuple(z.shape)}  xrec={tuple(xrec.shape)}  -> {out_path}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
