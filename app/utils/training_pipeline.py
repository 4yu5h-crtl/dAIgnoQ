from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import models

from dAIgnoQ.app import config
from dAIgnoQ.app.utils.quantum_utils import compute_states, get_quantum_device, kernel_from_states, scale_to_angles


class ClassicalTrainer:
    """Train a classical CNN classifier on prepared dataset splits."""

    def __init__(self, device: str = "cpu", checkpoints_dir: Optional[Path] = None):
        self.device = torch.device(device)
        self.checkpoints_dir = Path(checkpoints_dir or config.CHECKPOINTS_DIR)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def _accuracy(self, model: nn.Module, dataset, batch_size: int, img_size) -> float:
        loader = self._loader(dataset, batch_size=batch_size, shuffle=False)
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                imgs = F.interpolate(imgs, size=img_size, mode="bilinear", align_corners=False)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return float(correct / total) if total > 0 else 0.0

    def train(
        self,
        train_dataset,
        val_dataset=None,
        epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        img_size=(224, 224),
    ) -> Dict:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = self._loader(train_dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for _ in range(epochs):
            model.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                imgs = F.interpolate(imgs, size=img_size, mode="bilinear", align_corners=False)
                logits = model(imgs)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

        train_acc = self._accuracy(model, train_dataset, batch_size=batch_size, img_size=img_size)
        val_acc = self._accuracy(model, val_dataset, batch_size=batch_size, img_size=img_size) if val_dataset else None

        ckpt_path = self.checkpoints_dir / "classical_user_trained.pth"
        torch.save(model.state_dict(), ckpt_path)
        return {
            "checkpoint_path": str(ckpt_path),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "final_loss": losses[-1] if losses else None,
        }


class QuantumTrainer:
    """Train QSVM pipeline from image features using a quantum kernel."""

    def __init__(self, device: str = "cpu", checkpoints_dir: Optional[Path] = None):
        self.device = torch.device(device)
        self.checkpoints_dir = Path(checkpoints_dir or config.CHECKPOINTS_DIR)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _extract_features(self, dataset, batch_size: int, img_size=(224, 224)):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        backbone.to(self.device)
        backbone.eval()
        feats = []
        labels = []
        with torch.no_grad():
            for imgs, y in loader:
                imgs = imgs.to(self.device)
                imgs = F.interpolate(imgs, size=img_size, mode="bilinear", align_corners=False)
                f = backbone(imgs).cpu().numpy()
                feats.append(f)
                labels.append(y.numpy())
        X = np.concatenate(feats, axis=0)
        y = np.concatenate(labels, axis=0).astype(np.int64)
        return X, y, backbone

    def train_qsvm(
        self,
        train_dataset,
        val_dataset=None,
        pca_components: int = 12,
        n_qubits: int = 4,
        batch_size: int = 8,
    ) -> Dict:
        X_train, y_train, feature_backbone = self._extract_features(train_dataset, batch_size=batch_size)
        pca_n = max(2, min(pca_components, X_train.shape[0], X_train.shape[1]))
        pca = PCA(n_components=pca_n)
        X_train_pca = pca.fit_transform(X_train)

        mins = X_train_pca.min(axis=0)
        ranges = X_train_pca.max(axis=0) - mins
        ranges[ranges == 0] = 1e-8
        X_angles = scale_to_angles(X_train_pca, mins, ranges)

        q_device = get_quantum_device(n_qubits=n_qubits)
        states_train = compute_states(X_angles, n_qubits=n_qubits, device=q_device)
        K_train = kernel_from_states(states_train, states_train)

        svc = SVC(kernel="precomputed", probability=True)
        svc.fit(K_train, y_train)

        val_acc = None
        if val_dataset is not None:
            X_val, y_val, _ = self._extract_features(val_dataset, batch_size=batch_size)
            X_val_pca = pca.transform(X_val)
            X_val_angles = scale_to_angles(X_val_pca, mins, ranges)
            states_val = compute_states(X_val_angles, n_qubits=n_qubits, device=q_device)
            K_val = kernel_from_states(states_val, states_train)
            val_pred = svc.predict(K_val)
            val_acc = float((val_pred == y_val).mean())

        ckpt_path = self.checkpoints_dir / "qsvm_user_pipeline.pth"
        torch.save(
            {
                "svc": svc,
                "pca": pca,
                "mins": mins,
                "ranges": ranges,
                "states_train": states_train,
                "n_qubits": n_qubits,
            },
            ckpt_path,
        )

        return {
            "checkpoint_path": str(ckpt_path),
            "val_accuracy": val_acc,
            "n_qubits": n_qubits,
            "pca_components": pca_n,
        }
