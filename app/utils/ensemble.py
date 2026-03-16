"""
Ensemble / Decision Fusion Module for dAIgnoQ.

Fuses predictions from Classical (ResNet50) and Quantum (QSVM/VQC) models
into a single calibrated diagnosis, as described in the project architecture.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from PIL import Image


class EnsembleClassifier:
    """
    Meta-classifier that fuses outputs from multiple models (classical CNN + quantum)
    into a single prediction using configurable fusion strategies.
    """

    STRATEGIES = ["weighted_average", "max_confidence", "voting"]

    def __init__(self, classifiers: dict, strategy: str = "weighted_average", weights: Optional[dict] = None):
        """
        Args:
            classifiers: dict mapping model_name -> MedicalImageClassifier instance
                         e.g. {"resnet50": clf1, "qsvm": clf2}
            strategy:    fusion strategy — one of 'weighted_average', 'max_confidence', 'voting'
            weights:     optional dict mapping model_name -> float weight (must sum to 1.0)
                         Only used for 'weighted_average'. If None, equal weights are assigned.
        """
        self.classifiers = classifiers
        self.strategy = strategy

        if weights is None:
            n = len(classifiers)
            self.weights = {name: 1.0 / n for name in classifiers}
        else:
            self.weights = weights

    def predict(self, image: Image.Image) -> Tuple[int, float, Dict[str, dict]]:
        """
        Run all models and fuse their predictions.

        Returns:
            (final_class, final_confidence, individual_results)
            where individual_results maps model_name -> {"class": int, "confidence": float}
        """
        individual_results = {}

        for name, clf in self.classifiers.items():
            if clf.model is None:
                continue
            try:
                pred_class, confidence = clf.predict(image)
                individual_results[name] = {
                    "class": pred_class,
                    "confidence": float(confidence),
                }
            except Exception as e:
                individual_results[name] = {
                    "class": -1,
                    "confidence": 0.0,
                    "error": str(e),
                }

        if not individual_results:
            raise ValueError("No models produced a prediction. Load at least one model first.")

        # Filter out failed predictions
        valid = {k: v for k, v in individual_results.items() if v["class"] != -1}

        if not valid:
            raise ValueError("All models failed during prediction.")

        final_class, final_confidence = self._fuse(valid)

        return final_class, final_confidence, individual_results

    def _fuse(self, results: Dict[str, dict]) -> Tuple[int, float]:
        """Apply the configured fusion strategy."""
        if self.strategy == "weighted_average":
            return self._weighted_average(results)
        elif self.strategy == "max_confidence":
            return self._max_confidence(results)
        elif self.strategy == "voting":
            return self._majority_voting(results)
        else:
            return self._weighted_average(results)

    def _weighted_average(self, results: Dict[str, dict]) -> Tuple[int, float]:
        """
        Weighted average of confidence scores per class.
        Each model votes for its predicted class with weight * confidence.
        """
        from dAIgnoQ.app import config
        num_classes = len(config.CLASS_LABELS)
        class_scores = np.zeros(num_classes)

        total_weight = 0.0
        for name, res in results.items():
            w = self.weights.get(name, 1.0 / len(results))
            pred_class = res["class"]
            conf = res["confidence"]

            if 0 <= pred_class < num_classes:
                class_scores[pred_class] += w * conf
                total_weight += w

        if total_weight > 0:
            class_scores /= total_weight

        final_class = int(np.argmax(class_scores))
        final_confidence = float(class_scores[final_class])

        return final_class, final_confidence

    def _max_confidence(self, results: Dict[str, dict]) -> Tuple[int, float]:
        """Pick the prediction from the model with the highest confidence."""
        best_name = max(results, key=lambda k: results[k]["confidence"])
        best = results[best_name]
        return best["class"], best["confidence"]

    def _majority_voting(self, results: Dict[str, dict]) -> Tuple[int, float]:
        """Majority vote across models. Confidence = fraction of models agreeing."""
        from collections import Counter
        votes = [res["class"] for res in results.values()]
        counter = Counter(votes)
        winner, count = counter.most_common(1)[0]

        # Confidence = agreement ratio * average confidence of agreeing models
        agreeing_confs = [res["confidence"] for res in results.values() if res["class"] == winner]
        agreement_ratio = count / len(votes)
        avg_conf = np.mean(agreeing_confs)

        return winner, float(agreement_ratio * avg_conf)

    def get_report(self, final_class: int, final_confidence: float,
                   individual_results: Dict[str, dict]) -> str:
        """Generate a human-readable fusion report."""
        from dAIgnoQ.app import config

        lines = ["## 🔀 Ensemble Decision Report", ""]
        lines.append(f"**Strategy:** {self.strategy.replace('_', ' ').title()}")
        lines.append(f"**Final Prediction:** {config.CLASS_LABELS.get(final_class, 'Unknown')}")
        lines.append(f"**Fused Confidence:** {final_confidence:.2%}")
        lines.append("")
        lines.append("### Individual Model Results")
        lines.append("| Model | Prediction | Confidence |")
        lines.append("|-------|-----------|------------|")

        for name, res in individual_results.items():
            if res["class"] == -1:
                lines.append(f"| {name} | ❌ Error | — |")
            else:
                label = config.CLASS_LABELS.get(res["class"], "Unknown")
                lines.append(f"| {name} | {label} | {res['confidence']:.2%} |")

        # Agreement check
        valid_preds = [res["class"] for res in individual_results.values() if res["class"] != -1]
        if len(set(valid_preds)) == 1:
            lines.append("")
            lines.append("> ✅ **All models agree** on the diagnosis.")
        elif len(set(valid_preds)) > 1:
            lines.append("")
            lines.append("> ⚠️ **Models disagree** — review individual results carefully.")

        return "\n".join(lines)
