import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_dwt.functional import dwt2


def charbonnier_loss(prediction, target, epsilon=1e-3):
    diff = prediction - target
    return torch.sqrt(diff * diff + epsilon * epsilon)


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        kernel = self.create_gaussian_kernel(self.kernel_size, self.sigma)
        self.register_buffer("gaussian_kernel", kernel)

    @staticmethod
    def create_gaussian_kernel(size, sigma):
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        g = g[:, None] * g[None, :]
        g = g / g.sum()
        return g

    def forward(self, x):
        # x: (B,C,H,W)
        c = x.size(1)
        k = self.gaussian_kernel[None, None, :, :].repeat(c, 1, 1, 1)  # (C,1,ks,ks)
        pad = self.kernel_size // 2
        return F.conv2d(x, k, padding=pad, groups=c)


def compute_wavelet_loss(x, x_hat, wave="haar", eps=1e-3):
    """
    Author snippet pattern:
      _, x_LH, x_HL, x_HH = dwt2(x, "haar").unbind(dim=1)
    We'll do exactly that.
    """
    Wx = dwt2(x, wave)
    Wh = dwt2(x_hat, wave)

    LL, LH, HL, HH = Wx.unbind(dim=1)
    LLh, LHh, HLh, HHh = Wh.unbind(dim=1)

    loss_LH = charbonnier_loss(LHh, LH, epsilon=eps).mean()
    loss_HL = charbonnier_loss(HLh, HL, epsilon=eps).mean()
    loss_HH = charbonnier_loss(HHh, HH, epsilon=eps).mean()
    return (loss_LH + loss_HL + loss_HH) / 3.0


class LiteVAEAuthorLoss(nn.Module):
    """
    Matches LiteAutoencoderKL.loss API:
      forward(inputs, reconstructions, posterior, optimizer_idx, global_step, last_layer=None, split="train")
    """

    def __init__(
        self,
        rec_type="l2",
        rec_weight=1.0,
        wavelet_weight=0.0,
        gaussian_weight=0.0,
        gaussian_kernel_size=5,
        gaussian_sigma=1.0,
        wavelet="haar",
        eps=1e-3,
        kl_weight_max=0.0,
        anneal_steps=0,
    ):
        super().__init__()
        self.rec_type = rec_type.lower()
        self.rec_weight = float(rec_weight)

        self.wavelet_weight = float(wavelet_weight)
        self.gaussian_weight = float(gaussian_weight)
        self.wavelet = wavelet
        self.eps = float(eps)

        self.kl_weight_max = float(kl_weight_max)
        self.anneal_steps = int(anneal_steps)

        self.gaussian = None
        if self.gaussian_weight > 0:
            self.gaussian = GaussianFilter(kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)

    def get_trainable_autoencoder_parameters(self):
        return []

    def _rec_loss(self, x, x_hat):
        if self.rec_type in ("l2", "mse"):
            return F.mse_loss(x_hat, x, reduction="mean")
        if self.rec_type in ("l1",):
            return F.l1_loss(x_hat, x, reduction="mean")
        # default: charbonnier
        return charbonnier_loss(x_hat, x, epsilon=self.eps).mean()

    def _kl_weight(self, global_step: int) -> float:
        if self.kl_weight_max <= 0 or self.anneal_steps <= 0:
            return self.kl_weight_max
        t = min(max(global_step, 0) / float(self.anneal_steps), 1.0)
        return self.kl_weight_max * t

    def forward(
        self,
        inputs,
        reconstructions,
        posterior,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train",
    ):
        # AE-only setup: ignore discriminator branch
        if optimizer_idx != 0:
            zero = reconstructions.sum() * 0.0
            return zero, {f"{split}/disc_loss": zero.detach()}

        rec = self._rec_loss(inputs, reconstructions)

        g_loss = torch.tensor(0.0, device=inputs.device)
        if self.gaussian_weight > 0 and self.gaussian is not None:
            xg = self.gaussian(inputs)
            xhg = self.gaussian(reconstructions)
            g_loss = charbonnier_loss(xhg, xg, epsilon=self.eps).mean()

        w_loss = torch.tensor(0.0, device=inputs.device)
        if self.wavelet_weight > 0:
            # Important (author snippet): disable autocast for wavelet ops
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                w_loss = compute_wavelet_loss(
                    inputs.float().contiguous(),
                    reconstructions.float().contiguous(),
                    wave=self.wavelet,
                    eps=self.eps,
                )

        kl = posterior.kl().mean()
        kl_w = self._kl_weight(global_step)

        total = (
            self.rec_weight * rec
            + self.gaussian_weight * g_loss
            + self.wavelet_weight * w_loss
            + kl_w * kl
        )

        log_dict = {
            f"{split}/rec_loss": rec.detach(),
            f"{split}/gaussian_loss": g_loss.detach(),
            f"{split}/wavelet_loss": w_loss.detach(),
            f"{split}/kl_loss": kl.detach(),
            f"{split}/kl_weight": torch.tensor(kl_w, device=inputs.device),
            f"{split}/total_loss": total.detach(),
        }
        return total, log_dict
