import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import os
from contextlib import contextmanager

from olvae.modules.distributions import DiagonalGaussianDistribution
from olvae.util import instantiate_from_config, count_params
from olvae.modules.ema import LitEma

def disabled_train(self, mode=True):
    return self

class LiteAutoencoderKL(pl.LightningModule):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 lossconfig,
                 embed_dim,
                 use_quant=True,
                 use_ema=False,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 save_top_k=3,
                 freeze_latent_space=False,
                 monitor=None,
                 mon_mode=None,
                 disc_lr_ratio=1.0,
                 use_dyn_loss=False,
                 **kwargs
                 ):
        super().__init__()
        self.automatic_optimization = False # Crucial for GANs

        self.image_key = image_key
        self.use_ema = use_ema
        self.use_quant = use_quant
        self.embed_dim = embed_dim
        self.freeze_latent_space = freeze_latent_space
        self.disc_lr_ratio = disc_lr_ratio
        self.use_dyn_loss = use_dyn_loss

        # 1. Handle VAE vs AE latent dimensions
        if not use_quant:
            # For VAE without separate quantization, encoder must output Mu and LogVar
            encoder_config.params.latent_dim = 2 * embed_dim
        
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(lossconfig)

        # 2. Setup Quantization/Bottleneck
        z_channels = embed_dim # Default
        self.quantizer = torch.nn.Module()
        # Identity mapping if use_quant is False
        self.quantizer.quant_conv = torch.nn.Identity() if not use_quant else torch.nn.Conv2d(embed_dim, 2*embed_dim, 1)
        self.quantizer.post_quant_conv = torch.nn.Identity() if not use_quant else torch.nn.Conv2d(embed_dim, embed_dim, 1)

        if freeze_latent_space:
            self.freeze_latents()

        # 3. EMA Setup
        if self.use_ema:
            self.encoder_ema = LitEma(self.encoder)
            self.decoder_ema = LitEma(self.decoder)
            self.quantizer_ema = LitEma(self.quantizer)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def freeze_latents(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.encoder.train = disabled_train

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quantizer.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.quantizer.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3: x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def hold_disc_grads(self, freeze: bool):
        """Standard GAN gradient toggle to prevent leaking into AE step."""
        disc = getattr(self.loss, "discriminator", None)
        if disc is not None:
            for p in disc.parameters():
                p.requires_grad = not freeze

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opt_ae, opt_disc = opts if isinstance(opts, list) else (opts, None)

        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        # --- OPTIMIZER 0: AE STEP ---
        self.toggle_optimizer(opt_ae)
        self.hold_disc_grads(freeze=True) # Ensure Disc is dead
        
        aeloss, log_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step, 
                                  last_layer=self.get_last_layer(), split="train")
        
        self.manual_backward(aeloss)
        self.clip_gradients(opt_ae, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_ae.step()
        opt_ae.zero_grad()
        self.untoggle_optimizer(opt_ae)

        # --- OPTIMIZER 1: DISC STEP ---
        if opt_disc is not None:
            self.toggle_optimizer(opt_disc)
            self.hold_disc_grads(freeze=False) # Wake up Disc
            
            discloss, log_d = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                       last_layer=self.get_last_layer(), split="train")
            
            if discloss > 0: # Avoid backward on zero loss if disc hasn't started
                self.manual_backward(discloss)
                self.clip_gradients(opt_disc, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                opt_disc.step()
            
            opt_disc.zero_grad()
            self.untoggle_optimizer(opt_disc)
            
            self.log_dict(log_d, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        self.log_dict(log_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        
        # Validation AE Loss
        aeloss, log_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                  last_layer=self.get_last_layer(), split="val")
        
        # Validation Disc Loss
        discloss, log_d = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                   last_layer=self.get_last_layer(), split="val")
        
        self.log_dict(log_ae)
        self.log_dict(log_d)
        return log_ae

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params = list(self.encoder.parameters()) + \
                    list(self.decoder.parameters()) + \
                    list(self.quantizer.parameters())
        
        opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))
        
        disc_params = self.loss.get_trainable_parameters()
        if not disc_params:
            return opt_ae
            
        opt_disc = torch.optim.Adam(disc_params, lr=lr * self.disc_lr_ratio, betas=(0.5, 0.9))
        return [opt_ae, opt_disc]

    def get_last_layer(self):
        # Used for adaptive weight calculation in some GAN losses
        try:
            return self.decoder.conv_out.weight
        except:
            return None