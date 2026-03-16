from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dAIgnoQ.app import config
from dAIgnoQ.app.utils.architectures import Generator, Discriminator, DiffusionUNet


class GenerativeModelTrainer:
    """Trainer for GAN and diffusion models on small medical-image datasets."""

    def __init__(
        self,
        model_type: str = "gan",
        device: str = "cpu",
        checkpoints_dir: Optional[Path] = None,
    ):
        if model_type not in {"gan", "diffusion"}:
            raise ValueError("model_type must be 'gan' or 'diffusion'")
        self.model_type = model_type
        self.device = torch.device(device)
        self.checkpoints_dir = Path(checkpoints_dir or config.CHECKPOINTS_DIR)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def _prepare_loader(self, dataset, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def _train_gan(
        self,
        dataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        latent_dim: int,
        img_size: int,
    ) -> Dict:
        loader = self._prepare_loader(dataset, batch_size)
        generator = Generator(latent_dim=latent_dim, img_size=img_size).to(self.device)
        discriminator = Discriminator(img_size=img_size).to(self.device)

        criterion = nn.BCELoss()
        opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        g_losses = []
        d_losses = []
        for _ in range(epochs):
            for real_imgs, _ in loader:
                real_imgs = real_imgs.to(self.device)
                real_imgs = F.interpolate(real_imgs, size=(img_size, img_size), mode="bilinear", align_corners=False)
                real_imgs = real_imgs * 2.0 - 1.0  # [0, 1] -> [-1, 1]

                batch_len = real_imgs.size(0)
                valid = torch.ones(batch_len, 1, device=self.device)
                fake = torch.zeros(batch_len, 1, device=self.device)

                # Train generator
                opt_g.zero_grad()
                z = torch.randn(batch_len, latent_dim, device=self.device)
                gen_imgs = generator(z)
                g_loss = criterion(discriminator(gen_imgs), valid)
                g_loss.backward()
                opt_g.step()

                # Train discriminator
                opt_d.zero_grad()
                real_loss = criterion(discriminator(real_imgs), valid)
                fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2.0
                d_loss.backward()
                opt_d.step()

                g_losses.append(float(g_loss.item()))
                d_losses.append(float(d_loss.item()))

        ckpt_path = self.checkpoints_dir / "gan_generator_user_trained.pth"
        torch.save(generator.state_dict(), ckpt_path)
        self.model = generator
        return {
            "model_type": "gan",
            "checkpoint_path": str(ckpt_path),
            "g_loss": g_losses[-1] if g_losses else None,
            "d_loss": d_losses[-1] if d_losses else None,
            "steps": len(g_losses),
        }

    def _train_diffusion(
        self,
        dataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        img_size: int,
    ) -> Dict:
        from diffusers import DDPMScheduler

        loader = self._prepare_loader(dataset, batch_size)
        model = DiffusionUNet(sample_size=img_size).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

        losses = []
        for _ in range(epochs):
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                imgs = F.interpolate(imgs, size=(img_size, img_size), mode="bilinear", align_corners=False)
                imgs = imgs * 2.0 - 1.0
                noise = torch.randn_like(imgs)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (imgs.shape[0],), device=self.device
                ).long()
                noisy_imgs = scheduler.add_noise(imgs, noise, timesteps)
                pred_noise = model(noisy_imgs, timesteps)
                loss = F.mse_loss(pred_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

        ckpt_path = self.checkpoints_dir / "diffusion_unet_user_trained.pth"
        torch.save(model.model.state_dict(), ckpt_path)
        self.model = model
        return {
            "model_type": "diffusion",
            "checkpoint_path": str(ckpt_path),
            "loss": losses[-1] if losses else None,
            "steps": len(losses),
        }

    def train(
        self,
        dataset,
        epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        latent_dim: int = 100,
        img_size: Optional[int] = None,
    ) -> Dict:
        if self.model_type == "gan":
            return self._train_gan(
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                latent_dim=latent_dim,
                img_size=img_size or config.GAN_IMG_SIZE,
            )

        return self._train_diffusion(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            img_size=img_size or config.DIFFUSION_IMG_SIZE,
        )
