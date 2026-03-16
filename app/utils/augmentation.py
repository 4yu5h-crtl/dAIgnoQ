from typing import Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AugmentedDataset(Dataset):
    """Apply stochastic augmentations to an existing dataset."""

    def __init__(self, base_dataset: Dataset, augment_transform):
        self.base_dataset = base_dataset
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        if not isinstance(img, Image.Image):
            raise TypeError("Expected PIL image or tensor-compatible image.")
        aug = self.augment_transform(img)
        return aug, label


class DataAugmentor:
    """Build and apply configurable augmentation transforms."""

    @staticmethod
    def build_transform(config: Dict, output_size=(224, 224)):
        ops = [transforms.Resize(output_size)]
        if config.get("rotation"):
            ops.append(transforms.RandomRotation(degrees=float(config["rotation"])))
        if config.get("flip", False):
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        brightness = float(config.get("brightness", 0.0))
        contrast = float(config.get("contrast", 0.0))
        if brightness > 0 or contrast > 0:
            ops.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
        blur = float(config.get("blur", 0.0))
        if blur > 0:
            ops.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, blur)))
        ops.extend(
            [
                transforms.ToTensor(),
            ]
        )
        return transforms.Compose(ops)

    @classmethod
    def augment_train_split(cls, split_datasets: Dict, config: Dict, output_size=(224, 224)) -> Dict:
        """
        Apply augmentation to train split only and return updated split dict.
        Keeps val/test unchanged.
        """
        if "train" not in split_datasets:
            raise ValueError("Expected split_datasets with a 'train' key.")

        transform = cls.build_transform(config, output_size=output_size)
        augmented_train = AugmentedDataset(split_datasets["train"], transform)
        result = dict(split_datasets)
        result["train_augmented"] = augmented_train
        result["augmentation_config"] = dict(config)
        return result
