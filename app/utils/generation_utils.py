import torch
import numpy as np
from PIL import Image
from dAIgnoQ.app.utils.architectures import Generator, DiffusionUNet
from dAIgnoQ.app import config


def generate_gan_image(model_path, device='cpu', latent_dim=100, img_size=64):
    """
    Generate an image using a pre-trained GAN generator.
    """
    generator = Generator(latent_dim=latent_dim, img_size=img_size)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()

    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        fake_img = generator(z).cpu().squeeze(0)

    # Denormalize (assuming Tanh output [-1, 1])
    fake_img = (fake_img + 1) / 2
    fake_img = fake_img.clamp(0, 1).numpy().transpose(1, 2, 0)
    fake_img = (fake_img * 255).astype(np.uint8)

    return Image.fromarray(fake_img)


def generate_diffusion_image(model_path, device='cpu', img_size=128, num_inference_steps=50):
    """
    Generate an image using a DDPM diffusion sampling loop.

    Uses the diffusers DDPMScheduler for proper reverse-diffusion denoising.
    If a trained checkpoint exists at model_path, it loads the UNet from it.
    Otherwise, it runs with random initialization to demonstrate the pipeline.
    """
    try:
        from diffusers import DDPMScheduler

        # Build UNet
        unet = DiffusionUNet(sample_size=img_size)

        # Try loading trained weights
        if model_path is not None:
            try:
                state_dict = torch.load(model_path, map_location=device)
                unet.model.load_state_dict(state_dict)
            except Exception:
                pass  # Fall back to untrained model for demo

        unet.to(device)
        unet.eval()

        # Set up scheduler
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )
        scheduler.set_timesteps(num_inference_steps)

        # Start from pure noise
        sample = torch.randn(1, 3, img_size, img_size).to(device)

        # Reverse diffusion loop
        with torch.no_grad():
            for t in scheduler.timesteps:
                t_tensor = torch.tensor([t], device=device).long()
                model_output = unet(sample, t_tensor)
                sample = scheduler.step(model_output, int(t), sample).prev_sample

        # Convert to image
        sample = sample.cpu().squeeze(0)
        sample = (sample + 1) / 2  # [-1, 1] -> [0, 1]
        sample = sample.clamp(0, 1).numpy().transpose(1, 2, 0)
        sample = (sample * 255).astype(np.uint8)

        return Image.fromarray(sample)

    except ImportError:
        # diffusers not installed — return a placeholder
        placeholder = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        return Image.fromarray(placeholder)

    except Exception as e:
        # Any other error — return placeholder with error info print
        print(f"Diffusion generation error: {e}")
        placeholder = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        return Image.fromarray(placeholder)
