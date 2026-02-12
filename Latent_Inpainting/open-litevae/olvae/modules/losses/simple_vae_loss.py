import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAELoss(nn.Module):
    def __init__(self, rec_type="l2", rec_weight=0.8, kl_weight_max=1e-4, anneal_steps=5000):
        super().__init__()
        self.rec_type = rec_type.lower()
        self.rec_weight = float(rec_weight)
        self.kl_weight_max = float(kl_weight_max)
        self.anneal_steps = anneal_steps 

    def get_trainable_autoencoder_parameters(self):
        """
        Required by LiteAutoencoderKL to collect parameters for the optimizer.
        Since we have no learnable weights in this simple loss, we return an empty list.
        """
        return []

    def get_kl_weight(self, global_step):
        if self.anneal_steps <= 0:
            return self.kl_weight_max
        return min(self.kl_weight_max, self.kl_weight_max * (global_step / self.anneal_steps))

    def _reconstruction_loss(self, x, xrec):
        if self.rec_type in ["l2", "mse"]:
            return F.mse_loss(xrec, x, reduction="mean")
        return F.l1_loss(xrec, x, reduction="mean")

    def forward(
        self,
        inputs,
        reconstructions,
        posterior,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train", # This argument tells us if we are training or validating
          ):
        rec_loss = self._reconstruction_loss(inputs, reconstructions)
        kl_loss = posterior.kl().mean()
        
        # Use training step for annealing, but apply logic to both splits
        current_kl_weight = self.get_kl_weight(global_step)
        total = self.rec_weight * rec_loss + current_kl_weight * kl_loss

        # Use the 'split' variable to name keys (e.g., 'val/rec_loss' or 'train/rec_loss')
        log_dict = {
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/kl_loss": kl_loss.detach(),
            f"{split}/kl_weight": torch.tensor(current_kl_weight),
            f"{split}/total_loss": total.detach(),
        }
        
        return total, log_dict
# class SimpleVAELoss(nn.Module):
#     """
#     Minimal VAE loss compatible with LiteAutoencoderKL.

#     Uses:
#       total = rec_weight * rec_loss + kl_weight * kl_loss

#     Works with your current LiteAutoencoderKL training_step calling:
#       self.loss(inputs, reconstructions, posterior, 0, global_step, ...)
#     """

#     def __init__(self, rec_type="l1", rec_weight=1.0, kl_weight=1e-6):
#         super().__init__()
#         self.rec_type = rec_type.lower()
#         self.rec_weight = float(rec_weight)
#         self.kl_weight = float(kl_weight)

#     def get_trainable_autoencoder_parameters(self):
#         # Nothing extra besides encoder/decoder/quantizer
#         return []

#     def _reconstruction_loss(self, x, xrec):
#         if self.rec_type in ["l2", "mse"]:
#             return F.mse_loss(xrec, x, reduction="mean")
#         return F.l1_loss(xrec, x, reduction="mean")

#     def forward(
#         self,
#         inputs,
#         reconstructions,
#         posterior,
#         optimizer_idx,
#         global_step,
#         last_layer=None,
#         split="train",
#     ):
#         # We only support the AE step
#         rec_loss = self._reconstruction_loss(inputs, reconstructions)

#         kl_loss = posterior.kl().mean()

#         total = self.rec_weight * rec_loss + self.kl_weight * kl_loss
        
#         #debugging to find the cause of NaN and inf values in gradients
#         if torch.isnan(rec_loss) or torch.isinf(rec_loss):
#             raise RuntimeError("rec_loss is NaN/Inf")
#         if torch.isnan(kl_loss) or torch.isinf(kl_loss):
#             raise RuntimeError("kl_loss is NaN/Inf")
#         if torch.isnan(total) or torch.isinf(total):
#             raise RuntimeError("total loss is NaN/Inf")

#         log_dict = {
#             f"{split}/rec_loss": rec_loss.detach(),
#             f"{split}/kl_loss": kl_loss.detach(),
#             f"{split}/total_loss": total.detach(),
#         }
#         return total, log_dict

