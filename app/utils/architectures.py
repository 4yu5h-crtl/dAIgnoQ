import torch
import torch.nn as nn
from torchvision import models
from diffusers import UNet2DModel

# PennyLane import (for HybridModel)
try:
    import pennylane as qml
    from pennylane.qnn import TorchLayer
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

class HybridModel(nn.Module):
    """
    Hybrid CNN-VQC Model using ResNet18 as backbone.
    Extracted from CNN_WITH_VQC.ipynb.
    """
    def __init__(self, cnn_backbone=None, n_qubits=12, n_layers=3, num_classes=2, qml_layer=None):
        super().__init__()
        if cnn_backbone is None:
            # Default ResNet18 backbone
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            modules = list(resnet.children())[:-1]
            self.cnn = nn.Sequential(*modules)
        else:
            self.cnn = cnn_backbone
            
        self.compress = nn.Linear(512, n_qubits)
        self.vqc = qml_layer
        self.fc = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.cnn(x)         # [B,512,1,1]
        x = x.view(x.size(0), -1)  # [B,512]
        x = self.compress(x)    # [B,n_qubits]
        if self.vqc is not None:
            q = self.vqc(x)     # [B,n_qubits]
            out = self.fc(q)    # [B,num_classes]
        else:
            out = self.fc(x)    # Fallback if VQC is not provided
        return out

class ResNetVQC(HybridModel):
    """
    Alias for HybridModel.
    """
    pass

class Generator(nn.Module):
    """
    Generator for GAN.
    Extracted from GAN.ipynb.
    """
    def __init__(self, latent_dim=100, channels=3, img_size=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), self.channels, self.img_size, self.img_size)
        return img

class Discriminator(nn.Module):
    """
    Discriminator for GAN.
    Extracted from GAN.ipynb.
    """
    def __init__(self, channels=3, img_size=64):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class DiffusionUNet(nn.Module):
    """
    UNet wrapper for Diffusion model using diffusers library.
    Extracted from Diffusion.ipynb.
    """
    def __init__(self, sample_size=128, in_channels=3, out_channels=3):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            )
        )

    def forward(self, x, timestep):
        return self.model(x, timestep).sample
