import torch
from torch import nn, cuda
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torchvision
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torchaudio

# Optional: try keeping cuDNN enabled; if you hit cuDNN errors, uncomment the next line
# torch.backends.cudnn.enabled = False

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#original partial conv2d layer from NVIDIA



print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("Torchaudio:", torchaudio.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version PyTorch was compiled with:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
else:
    print("No GPU detected by PyTorch")


