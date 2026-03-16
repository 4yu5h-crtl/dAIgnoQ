from typing import Any, Dict, List

import cv2
import numpy as np
import torch


class DataIntelligenceEngine:
    """Heuristic engine for auto-selecting generation and augmentation strategy."""

    @staticmethod
    def _to_numpy_image(sample: Any) -> np.ndarray:
        if isinstance(sample, tuple):
            img = sample[0]
        else:
            img = sample

        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
            return arr

        if hasattr(img, "convert"):  # PIL image
            return np.array(img.convert("RGB"))

        raise TypeError("Unsupported image type for intelligence analysis.")

    @classmethod
    def _sample_quality_metrics(cls, dataset, max_samples: int = 64) -> Dict[str, float]:
        total = len(dataset)
        if total == 0:
            return {"mean_edge_density": 0.0, "mean_blur_score": 0.0, "mean_diversity": 0.0}

        sample_count = min(total, max_samples)
        indices = np.linspace(0, total - 1, num=sample_count, dtype=int)
        edge_scores: List[float] = []
        blur_scores: List[float] = []
        flat_vectors: List[np.ndarray] = []

        for idx in indices:
            arr = cls._to_numpy_image(dataset[idx])
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edge_scores.append(float((edges > 0).mean()))
            blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            flat_vectors.append(small.flatten())

        mat = np.stack(flat_vectors, axis=0)
        diversity = float(np.mean(np.std(mat, axis=0)))

        return {
            "mean_edge_density": float(np.mean(edge_scores)),
            "mean_blur_score": float(np.mean(blur_scores)),
            "mean_diversity": diversity,
        }

    @classmethod
    def analyze_and_recommend(cls, dataset_info: Dict[str, Any], train_dataset) -> Dict[str, Any]:
        total_images = int(dataset_info.get("total_images", len(train_dataset)))
        pos_images = int(dataset_info.get("positive_images", 0))
        neg_images = int(dataset_info.get("negative_images", 0))
        pos_ratio = float(dataset_info.get("positive_ratio", 0.5))
        imbalance = abs(pos_ratio - 0.5) * 2.0

        quality = cls._sample_quality_metrics(train_dataset)
        blur = quality["mean_blur_score"]
        diversity = quality["mean_diversity"]
        edge_density = quality["mean_edge_density"]

        reasons = []

        if total_images < 300:
            generator_model = "gan"
            reasons.append("Small dataset detected; GAN is preferred for low-data convergence speed.")
        elif diversity > 0.16 and total_images >= 300:
            generator_model = "diffusion"
            reasons.append("Dataset has higher visual diversity; diffusion is preferred for richer synthesis.")
        else:
            generator_model = "gan"
            reasons.append("Moderate dataset complexity; GAN chosen for efficient synthesis.")

        use_augmentation = False
        aug_config = {"rotation": 0, "flip": False, "brightness": 0.0, "contrast": 0.0, "blur": 0.0}

        if total_images < 500 or imbalance > 0.25:
            use_augmentation = True
            aug_config.update({"rotation": 12, "flip": True, "brightness": 0.1, "contrast": 0.1})
            reasons.append("Limited/imbalanced data; augmentation recommended to improve generalization.")

        if blur < 50:
            use_augmentation = True
            aug_config["contrast"] = max(aug_config["contrast"], 0.15)
            reasons.append("Images appear soft/blurry; slight contrast augmentation recommended.")

        if edge_density < 0.06:
            use_augmentation = True
            aug_config["rotation"] = max(aug_config["rotation"], 15)
            reasons.append("Low edge density suggests weak structural variation; rotation augmentation recommended.")

        synthetic_multiplier = 3 if total_images < 500 else 2
        synthetic_target = max(200, total_images * synthetic_multiplier)

        if generator_model == "gan":
            train_defaults = {"epochs": 10 if total_images < 300 else 20, "batch_size": 8, "learning_rate": 2e-4}
        else:
            train_defaults = {"epochs": 8 if total_images < 300 else 15, "batch_size": 4, "learning_rate": 1e-4}

        return {
            "generator_model": generator_model,
            "use_augmentation": use_augmentation,
            "augmentation_config": aug_config,
            "synthetic_target_count": int(synthetic_target),
            "train_defaults": train_defaults,
            "quality_metrics": quality,
            "dataset_summary": {
                "total_images": total_images,
                "positive_images": pos_images,
                "negative_images": neg_images,
                "imbalance_score": imbalance,
            },
            "reasons": reasons,
        }
