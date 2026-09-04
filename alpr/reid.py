"""
Vehicle Re-Identification (ReID) module for multi-camera vehicle tracking.

Phase 5A: Baseline visual feature extractor using ImageNet-pretrained ResNet-18
producing strict L2-normalized 512-D embeddings and cosine similarity metric.
Phase 5B: Drop-in support for vehicle-specific weights (VeRi-776 / VehicleID).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

log = logging.getLogger("alpr.reid")


class VehicleReID:
    """
    Vehicle appearance feature extractor and metric comparison engine.

    Extracts fixed-dimension (512-D), L2-normalized visual appearance vectors
    from vehicle crops for cross-observation re-identification.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        backbone: str = "resnet18",
        device: str = "cpu",
        img_size: Tuple[int, int] = (224, 224),  # (height, width)
    ):
        """
        Initialize the ReID feature extractor.

        Args:
            weights_path: Path to custom vehicle-specific weights (.pth / .pt).
                          If None, uses ImageNet pretrained ResNet-18 baseline (Phase 5A).
            backbone: Model architecture name (default: "resnet18").
            device: Compute device ('cpu', 'cuda', etc.).
            img_size: Input resolution (height, width).
        """
        self.backbone_name = backbone
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.img_size = img_size  # (H, W)
        self.weights_path = Path(weights_path) if weights_path else None

        # Normalization constants (ImageNet standards)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.model, self.embedding_dim = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        mode_label = f"Phase 5B ({self.weights_path.name})" if self.weights_path else "Phase 5A baseline (ImageNet ResNet-18)"
        log.info("VehicleReID initialized: %s | Dim: %d | Device: %s", mode_label, self.embedding_dim, self.device)

    def _load_model(self) -> Tuple[nn.Module, int]:
        """Build feature extractor backbone and remove classification head."""
        if self.backbone_name == "resnet18":
            if self.weights_path and self.weights_path.exists():
                model = torchvision.models.resnet18(weights=None)
                state_dict = torch.load(str(self.weights_path), map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                log.info("Loaded custom ReID weights from %s", self.weights_path)
            else:
                model = torchvision.models.resnet18(
                    weights=torchvision.models.ResNet18_Weights.DEFAULT
                )
            feature_dim = model.fc.in_features  # 512
            model.fc = nn.Identity()
            return model, feature_dim
        else:
            raise ValueError(f"Unsupported ReID backbone: {self.backbone_name}")

    def preprocess_crop(self, crop: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess a vehicle BGR crop into a normalized torch Tensor.

        Returns:
            Tensor of shape (1, 3, H, W) or None if crop is invalid.
        """
        if (
            crop is None
            or not isinstance(crop, np.ndarray)
            or crop.size == 0
            or crop.shape[0] < 10
            or crop.shape[1] < 10
        ):
            return None

        # BGR -> RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Resize to model input size (width, height for cv2.resize)
        h, w = self.img_size
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize: (x / 255.0 - mean) / std
        norm_img = (resized.astype(np.float32) / 255.0 - self.mean) / self.std

        # HWC -> CHW -> BCHW
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    def extract_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a strictly L2-normalized feature embedding from a single vehicle crop.

        Args:
            crop: BGR vehicle image as a numpy array.

        Returns:
            1D float32 numpy array of shape (embedding_dim,) with ||e||_2 == 1.0,
            or None if crop is invalid.
        """
        tensor = self.preprocess_crop(crop)
        if tensor is None:
            return None

        tensor = tensor.to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = F.normalize(feat, p=2, dim=1)

        embedding = feat.squeeze(0).cpu().numpy().astype(np.float32)
        return embedding

    def extract_batch(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Batch inference for multiple vehicle crops to maximize CPU/GPU efficiency.

        Args:
            crops: List of BGR crop images.

        Returns:
            List of embeddings matching the input length. Elements are None for invalid crops.
        """
        if not crops:
            return []

        results: List[Optional[np.ndarray]] = [None] * len(crops)
        valid_indices = []
        valid_tensors = []

        for idx, crop in enumerate(crops):
            t = self.preprocess_crop(crop)
            if t is not None:
                valid_indices.append(idx)
                valid_tensors.append(t)

        if not valid_tensors:
            return results

        # Stack into single batch tensor: (B, 3, H, W)
        batch = torch.cat(valid_tensors, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
            feats = F.normalize(feats, p=2, dim=1)

        feats_np = feats.cpu().numpy().astype(np.float32)
        for i, original_idx in enumerate(valid_indices):
            results[original_idx] = feats_np[i]

        return results

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature embeddings.

        Args:
            emb1: Feature vector 1.
            emb2: Feature vector 2.

        Returns:
            Cosine similarity clamped to [-1.0, 1.0]. Returns 0.0 if vectors are invalid.
        """
        if emb1 is None or emb2 is None:
            return 0.0

        v1 = np.asarray(emb1, dtype=np.float32).ravel()
        v2 = np.asarray(emb2, dtype=np.float32).ravel()

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        dot = float(np.dot(v1, v2))
        cosine = dot / (norm1 * norm2)
        return max(-1.0, min(1.0, float(cosine)))

    @staticmethod
    def aggregate_embeddings(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Mean-pool multiple representative crop embeddings and L2 re-normalize
        into a single stable track-level embedding.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            L2-normalized aggregate embedding of shape (embedding_dim,),
            or None if no valid embeddings provided.
        """
        valid = [e for e in embeddings if e is not None and len(e) > 0]
        if not valid:
            return None

        # Mean pooling
        stacked = np.stack(valid, axis=0)
        mean_vec = np.mean(stacked, axis=0)

        # L2 re-normalization
        norm = np.linalg.norm(mean_vec)
        if norm < 1e-6:
            return None

        unit_vec = (mean_vec / norm).astype(np.float32)
        return unit_vec


# Module-level convenience functions
_default_reid_engine: Optional[VehicleReID] = None


def get_default_reid_engine() -> VehicleReID:
    """Get or instantiate the singleton default ReID engine."""
    global _default_reid_engine
    if _default_reid_engine is None:
        _default_reid_engine = VehicleReID()
    return _default_reid_engine


def extract_embedding(crop: np.ndarray) -> Optional[np.ndarray]:
    """Extract L2-normalized 512-D embedding from vehicle crop using default engine."""
    return get_default_reid_engine().extract_embedding(crop)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings in [-1.0, 1.0]."""
    return VehicleReID.compute_similarity(emb1, emb2)


def aggregate_embeddings(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    """Mean-pool and L2 re-normalize a list of embeddings."""
    return VehicleReID.aggregate_embeddings(embeddings)
