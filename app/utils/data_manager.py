"""
Data Management Module for dAIgnoQ

Handles dataset loading, format detection, validation, and splitting.
Supports 3 upload formats:
  - Format A: Organized subfolders (positive/ and negative/)
  - Format B: Images folder + CSV labels file
  - Format C: Single folder (user specifies label ratio)
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class MedicalImageDataset(Dataset):
    """PyTorch Dataset for medical images with labels."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of binary labels (0 or 1)
            transform: Optional torchvision transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        if len(image_paths) != len(labels):
            raise ValueError("Number of images must match number of labels")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Returns image tensor and label."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default: convert to tensor
                image = torch.from_numpy(np.array(image)).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC -> CHW
            
            return image, label
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")


class DatasetManager:
    """
    Manages dataset loading, format detection, validation, and splitting.
    """
    
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    BATCH_SIZE = 32
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize DatasetManager.
        
        Args:
            base_path: Base path for data directory (default: None, use current)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.current_dataset = None
        self.current_labels = None
        self.dataset_info = {}
    
    @staticmethod
    def detect_format(folder_path: str) -> str:
        """
        Detect the format of uploaded dataset.
        
        Returns:
            'folders' - Format A: positive/ and negative/ subfolders
            'csv'     - Format B: images/ folder + labels.csv
            'single'  - Format C: single folder with images
            'unknown' - Could not detect format
        """
        path = Path(folder_path)
        
        if not path.exists() or not path.is_dir():
            return 'unknown'
        
        contents = list(path.iterdir())
        folder_names = [item.name.lower() for item in contents if item.is_dir()]
        file_names = [item.name.lower() for item in contents if item.is_file()]
        
        # Check for Format A: positive/ and negative/ folders
        if ('positive' in folder_names or 'pos' in folder_names) and \
           ('negative' in folder_names or 'neg' in folder_names):
            return 'folders'
        
        # Check for Format B: images/ folder + labels.csv
        if 'images' in folder_names and any(f.endswith('.csv') for f in file_names):
            return 'csv'
        
        # Check for Format C: single folder with images
        image_files = [f for f in file_names if any(f.endswith(fmt) for fmt in DatasetManager.SUPPORTED_FORMATS)]
        if image_files:
            return 'single'
        
        # Check if subfolders contain images
        for item in contents:
            if item.is_dir():
                sub_files = list(item.iterdir())
                sub_images = [f for f in sub_files if f.is_file() and 
                            any(f.name.endswith(fmt) for fmt in DatasetManager.SUPPORTED_FORMATS)]
                if sub_images:
                    return 'folders'
        
        return 'unknown'
    
    def load_dataset_folders(self, folder_path: str) -> Tuple[List[str], List[int], Dict]:
        """
        Load dataset from Format A (positive/ and negative/ subfolders).
        
        Returns:
            (image_paths, labels, stats_dict)
        """
        path = Path(folder_path)
        image_paths = []
        labels = []
        stats = {'total': 0, 'positive': 0, 'negative': 0}
        
        # Find positive folder
        pos_folders = [p for p in path.iterdir() if p.is_dir() and p.name.lower() in ('positive', 'pos')]
        if pos_folders:
            pos_path = pos_folders[0]
            for img_file in pos_path.iterdir():
                if img_file.suffix.lower() in self.SUPPORTED_FORMATS:
                    image_paths.append(str(img_file))
                    labels.append(1)
                    stats['positive'] += 1
        
        # Find negative folder
        neg_folders = [p for p in path.iterdir() if p.is_dir() and p.name.lower() in ('negative', 'neg')]
        if neg_folders:
            neg_path = neg_folders[0]
            for img_file in neg_path.iterdir():
                if img_file.suffix.lower() in self.SUPPORTED_FORMATS:
                    image_paths.append(str(img_file))
                    labels.append(0)
                    stats['negative'] += 1
        
        stats['total'] = len(image_paths)
        
        if stats['total'] == 0:
            raise ValueError("No images found in positive/ or negative/ folders")
        
        return image_paths, labels, stats
    
    def load_dataset_csv(self, folder_path: str) -> Tuple[List[str], List[int], Dict]:
        """
        Load dataset from Format B (images/ folder + labels.csv).
        
        CSV format: image_filename, label
        (where label is 0 for negative, 1 for positive)
        
        Returns:
            (image_paths, labels, stats_dict)
        """
        path = Path(folder_path)
        images_path = path / 'images'
        csv_file = None
        
        # Find CSV file
        for file in path.iterdir():
            if file.suffix.lower() == '.csv':
                csv_file = file
                break
        
        if not csv_file:
            raise ValueError("No CSV file found in dataset folder")
        
        if not images_path.exists():
            raise ValueError("No 'images' folder found in dataset")
        
        image_paths = []
        labels = []
        stats = {'total': 0, 'positive': 0, 'negative': 0}
        
        # Read CSV
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header if present
            
            for row in reader:
                if len(row) < 2:
                    continue
                
                filename = row[0].strip()
                raw_label = row[1].strip().lower()
                if raw_label in {"1", "positive", "pos", "disease", "glaucoma"}:
                    label = 1
                elif raw_label in {"0", "negative", "neg", "healthy", "normal"}:
                    label = 0
                else:
                    raise ValueError(
                        f"Invalid label '{row[1]}' in {csv_file.name}. "
                        "Use 0/1 or negative/positive style labels."
                    )
                
                img_path = images_path / filename
                if img_path.exists():
                    image_paths.append(str(img_path))
                    labels.append(label)
                    stats['total'] += 1
                    stats['positive' if label == 1 else 'negative'] += 1
        
        if stats['total'] == 0:
            raise ValueError("No valid image-label pairs found in CSV")
        
        return image_paths, labels, stats
    
    def load_dataset_single(self, folder_path: str, positive_ratio: float = 0.5) -> Tuple[List[str], List[int], Dict]:
        """
        Load dataset from Format C (single folder with images).
        User specifies what ratio of images should be labeled as positive.
        
        Args:
            folder_path: Path to folder with images
            positive_ratio: Ratio of images to label as positive (0.0-1.0)
        
        Returns:
            (image_paths, labels, stats_dict)
        """
        path = Path(folder_path)
        image_paths = []
        
        # Collect all images
        for item in path.rglob('*'):
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_FORMATS:
                image_paths.append(str(item))
        
        if len(image_paths) == 0:
            raise ValueError("No images found in folder")
        
        # Sort for reproducibility
        image_paths.sort()
        
        # Create labels based on positive_ratio
        num_positive = int(len(image_paths) * positive_ratio)
        labels = [1] * num_positive + [0] * (len(image_paths) - num_positive)
        
        # Shuffle to mix positive and negative
        indices = np.random.permutation(len(image_paths))
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        stats = {
            'total': len(image_paths),
            'positive': sum(labels),
            'negative': len(labels) - sum(labels)
        }
        
        return image_paths, labels, stats
    
    def load_dataset(self, folder_path: str, positive_ratio: Optional[float] = None) -> Tuple[MedicalImageDataset, Dict]:
        """
        Load dataset from any supported format.
        
        Args:
            folder_path: Path to dataset folder
            positive_ratio: For Format C, ratio of positive labels (0.0-1.0)
                          If None, defaults to 0.5
        
        Returns:
            (MedicalImageDataset, stats_dict)
        """
        format_type = self.detect_format(folder_path)
        
        if format_type == 'unknown':
            raise ValueError(f"Could not detect dataset format in {folder_path}")
        
        # Load based on format
        if format_type == 'folders':
            image_paths, labels, stats = self.load_dataset_folders(folder_path)
        elif format_type == 'csv':
            image_paths, labels, stats = self.load_dataset_csv(folder_path)
        elif format_type == 'single':
            if positive_ratio is None:
                positive_ratio = 0.5
            image_paths, labels, stats = self.load_dataset_single(folder_path, positive_ratio)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        # Store for later use
        self.current_dataset = image_paths
        self.current_labels = labels
        self.dataset_info = {
            'format': format_type,
            'total_images': stats['total'],
            'positive_images': stats['positive'],
            'negative_images': stats['negative'],
            'positive_ratio': stats['positive'] / stats['total'] if stats['total'] > 0 else 0
        }
        
        # Create PyTorch Dataset
        dataset = MedicalImageDataset(image_paths, labels)
        
        return dataset, self.dataset_info
    
    def split_dataset(
        self,
        dataset: MedicalImageDataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Dict[str, MedicalImageDataset]:
        """
        Split dataset into train/val/test.
        
        Args:
            dataset: MedicalImageDataset to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Seed for reproducibility
        
        Returns:
            Dictionary with 'train', 'val', 'test' datasets
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        if len(dataset) < 3:
            raise ValueError("Dataset must contain at least 3 images for train/val/test split.")
        
        # Set seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        dataset_size = len(dataset)
        train_size = max(1, int(train_ratio * dataset_size))
        val_size = max(1, int(val_ratio * dataset_size))
        test_size = dataset_size - train_size - val_size
        if test_size < 1:
            test_size = 1
            if train_size > val_size:
                train_size -= 1
            else:
                val_size -= 1
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'sizes': {
                'train': train_size,
                'val': val_size,
                'test': test_size
            }
        }
    
    def get_dataloaders(
        self,
        split_datasets: Dict,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = True
    ) -> Dict:
        """
        Create DataLoaders from split datasets.
        
        Args:
            split_datasets: Output from split_dataset()
            batch_size: Batch size for training
            num_workers: Number of worker threads
            shuffle_train: Whether to shuffle training data
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        train_loader = DataLoader(
            split_datasets['train'],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            split_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            split_datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def validate_dataset(self, folder_path: str) -> Dict:
        """
        Validate dataset without loading it.
        
        Returns:
            Validation results dictionary
        """
        try:
            format_type = self.detect_format(folder_path)
            
            results = {
                'valid': True,
                'format': format_type,
                'message': f'Valid {format_type} format dataset'
            }
            
            if format_type == 'folders':
                image_paths, labels, stats = self.load_dataset_folders(folder_path)
            elif format_type == 'csv':
                image_paths, labels, stats = self.load_dataset_csv(folder_path)
            elif format_type == 'single':
                image_paths, labels, stats = self.load_dataset_single(folder_path)
            else:
                results['valid'] = False
                results['message'] = 'Unknown format'
                return results
            
            results['stats'] = stats
            return results
        
        except Exception as e:
            return {
                'valid': False,
                'message': f'Validation failed: {str(e)}'
            }


def main():
    """Demo usage of DatasetManager."""
    # Example usage
    manager = DatasetManager()
    
    # Try to load a dataset
    try:
        dataset, stats = manager.load_dataset('path/to/dataset')
        print(f"Dataset loaded: {stats}")
        
        # Split dataset
        splits = manager.split_dataset(dataset)
        print(f"Split sizes: {splits['sizes']}")
        
        # Get dataloaders
        loaders = manager.get_dataloaders(splits, batch_size=32)
        print(f"DataLoaders created: {list(loaders.keys())}")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
