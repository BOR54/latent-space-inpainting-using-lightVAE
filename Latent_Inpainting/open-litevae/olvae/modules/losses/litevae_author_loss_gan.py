import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torch_dwt.functional import dwt2
from olvae.modules.unetgan.d_unet import Unet_Discriminator

# ----------------------------
# GAN loss functions (Keep as helper functions)
# ----------------------------
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)

def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)

def get_gan_loss(name: str):
    if name.lower() == "hinge":
        return hinge_g_loss, hinge_d_loss
    raise ValueError(f"Unsupported disc_loss={name}")

def g_loss_wrappper(logits_fake, loss_fn):
    if isinstance(logits_fake, torch.Tensor):
        return loss_fn(logits_fake)
    loss = 0.0
    for fi in logits_fake:
        loss += loss_fn(fi)
    return loss / float(len(logits_fake))

def d_loss_wrappper(logits_real, logits_fake, loss_fn):
    if isinstance(logits_real, torch.Tensor):
        return loss_fn(logits_real, logits_fake)
    loss = 0.0
    for ri, fi in zip(logits_real, logits_fake):
        loss += loss_fn(ri, fi)
    return loss / float(len(logits_real))

def charbonnier_loss(prediction, target, epsilon=1e-3):
    return torch.sqrt((prediction - target)**2 + epsilon**2)

# ----------------------------
# Main Module
# ----------------------------
class LiteVAEAuthorGANLoss(nn.Module):
    def __init__(
        self,
        rec_type="l1",
        rec_weight=1.0,
        wavelet_weight=0.0,
        gaussian_weight=0.0,
        kl_weight_max=0.0,
        anneal_steps=0,
        use_gan=True,
        disc_loss="hinge",
        disc_weight=0.1,
        disc_start=0,
        disc_ch=64,
        disc_resolution=256,
        disc_attn="64",
        eps=1e-3,
        wave="haar" # Added to params
    ):
        super().__init__()
        self.rec_type = rec_type.lower()
        self.rec_weight = rec_weight
        self.wavelet_weight = wavelet_weight
        self.gaussian_weight = gaussian_weight
        self.kl_weight_max = kl_weight_max
        self.anneal_steps = anneal_steps
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.use_gan = use_gan
        self.eps = eps
        self.wave = wave

        g_func, d_func = get_gan_loss(disc_loss)
        self.gen_loss = partial(g_loss_wrappper, loss_fn=g_func)
        self.disc_loss = partial(d_loss_wrappper, loss_fn=d_func)

        if self.use_gan:
            self.discriminator = Unet_Discriminator(
                D_ch=disc_ch,
                resolution=disc_resolution,
                D_attn=disc_attn,
                unconditional=True
            )
        else:
            self.discriminator = None

    def compute_wavelet_loss(self, x, x_hat):
        """
        Calculates loss on high-frequency wavelet sub-bands.
        """
        # torch_dwt.functional.dwt2 returns [B, 4*C, H/2, W/2]
        Wx = dwt2(x, self.wave)
        Wh = dwt2(x_hat, self.wave)
        
        c = x.size(1)
        # Split into [LL, LH, HL, HH] each with 'c' channels
        Wx_split = torch.split(Wx, c, dim=1)
        Wh_split = torch.split(Wh, c, dim=1)
        
        # Verify we have all 4 sub-bands to avoid IndexError
        if len(Wh_split) < 4:
            return torch.tensor(0.0, device=x.device)

        # LH, HL, HH components
        l1 = charbonnier_loss(Wh_split[1], Wx_split[1], epsilon=self.eps)
        l2 = charbonnier_loss(Wh_split[2], Wx_split[2], epsilon=self.eps)
        l3 = charbonnier_loss(Wh_split[3], Wx_split[3], epsilon=self.eps)
        
        total_wv = (l1 + l2 + l3) / 3.0
        return total_wv.mean()

    def _kl_weight(self, step):
        if self.anneal_steps <= 0: return self.kl_weight_max
        return self.kl_weight_max * min(1.0, step / self.anneal_steps)

    def forward(self, inputs, reconstructions, posterior, optimizer_idx, global_step, last_layer=None, split="train", **kwargs):
        # OPTIMIZER 0: AE + KL + GEN-ADV
        if optimizer_idx == 0:
            # Reconstruction
            if self.rec_type == "l1":
                rec = F.l1_loss(reconstructions, inputs)
            else:
                rec = F.mse_loss(reconstructions, inputs)

            # Wavelet Loss
            w_loss = torch.tensor(0.0, device=inputs.device)
            if self.wavelet_weight > 0:
                # Ensure float32 for DWT stability
                w_loss = self.compute_wavelet_loss(inputs.float(), reconstructions.float())

            # KL Loss
            kl = posterior.kl().mean()
            kl_w = self._kl_weight(global_step)

            # Generator Adversarial
            adv = torch.tensor(0.0, device=inputs.device)
            if self.use_gan and global_step >= self.disc_start:
                logits_fake = self.discriminator(reconstructions)
                adv = self.gen_loss(logits_fake)

            total = (self.rec_weight * rec) + (self.wavelet_weight * w_loss) + (kl_w * kl) + (self.disc_weight * adv)

            return total, {
                f"{split}/rec_loss": rec.detach(),
                f"{split}/kl_loss": kl.detach(),
                f"{split}/total_loss": total.detach(),
                f"{split}/g_adv": adv.detach(),
                f"{split}/wavelet_loss": w_loss.detach()
            }

        # OPTIMIZER 1: DISCRIMINATOR
        if optimizer_idx == 1:
            if not self.use_gan or global_step < self.disc_start:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True), {}

            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())
            d_loss = self.disc_loss(logits_real, logits_fake)

            return d_loss, {f"{split}/disc_loss": d_loss.detach()}

    def get_trainable_autoencoder_parameters(self):
          return []
  
    def get_trainable_parameters(self):
          if self.use_gan and self.discriminator is not None:
              return self.discriminator.parameters()
          return []