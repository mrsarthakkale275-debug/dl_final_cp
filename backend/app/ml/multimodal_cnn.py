"""Modular PyTorch implementation of the multimodal industrial inspection CNN.

Includes:
  - RGBAnomlayCNN:      WideResNet50-based feature extractor + anomaly head
  - XRayCNN:            ResNet34 backbone for grayscale radiographic inspection
  - GatedFusionModule:  Late cross-modal gated attention fusion
  - GradCAMGenerator:   Grad-CAM heatmap computation for any CNN layer
  - PatchCoreFeatureBank: Memory bank of normal patch features for unseen anomaly detection

All models include a mock/fallback mode so the server runs out-of-the-box
before custom training. When no trained weights are present, models use
pretrained ImageNet features and generate synthetic anomaly scores.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# ---------------------------------------------------------------------------
# RGB CNN: WideResNet50 backbone + dual-head anomaly scorer
# ---------------------------------------------------------------------------

class RGBAnomalyCNN(nn.Module):
    """WideResNet50-based RGB anomaly detection CNN.

    Dual heads:
      1. Anomaly scorer:  Global average pooling → FC → sigmoid → S_rgb ∈ [0,1]
      2. Feature extractor: Multi-scale feature maps for heatmap/localization

    In mock mode (no trained weights), uses pretrained ImageNet WideResNet50
    and a randomly-initialized classification head.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
            if pretrained:
                self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            else:
                self.backbone = wide_resnet50_2(weights=None)
        except Exception:
            from torchvision.models import wide_resnet50_2
            self.backbone = wide_resnet50_2(pretrained=pretrained)

        # Remove original FC
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Anomaly scoring head
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Classification head (normal vs defect types)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Store intermediate features for Grad-CAM
        self._features: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}

        # Register hooks on layer4 for Grad-CAM
        self.backbone.layer4.register_forward_hook(self._save_features("layer4"))

    def _save_features(self, name: str):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def _save_gradients(self, name: str):
        def hook(grad):
            self._gradients[name] = grad
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features through backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # Hook captures this

        # Register gradient hook for Grad-CAM
        if x.requires_grad:
            x.register_hook(self._save_gradients("layer4"))

        # Global average pool
        features = F.adaptive_avg_pool2d(x, 1).flatten(1)

        anomaly_score = self.anomaly_head(features)
        class_logits = self.classifier(features)
        class_probs = F.softmax(class_logits, dim=1)

        return {
            "anomaly_score": anomaly_score,
            "class_logits": class_logits,
            "class_probs": class_probs,
            "features": features,
            "feature_map": self._features.get("layer4"),
        }

    def extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale feature maps for PatchCore-style anomaly detection."""
        features = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        features.append(x)
        x = self.backbone.layer2(x)
        features.append(x)
        x = self.backbone.layer3(x)
        features.append(x)
        x = self.backbone.layer4(x)
        features.append(x)

        return features


# ---------------------------------------------------------------------------
# X-Ray CNN: ResNet34 backbone for grayscale radiographic inspection
# ---------------------------------------------------------------------------

class XRayCNN(nn.Module):
    """ResNet34-based X-ray internal inspection CNN.

    Adapted for single-channel grayscale radiograph input.
    Detects: voids, cracks, foreign inclusions, porosity.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            if pretrained:
                self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet34(weights=None)
        except Exception:
            from torchvision.models import resnet34
            self.backbone = resnet34(pretrained=pretrained)

        # Adapt first conv for single-channel input (average RGB weights)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Defect detection head
        self.defect_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Anomaly scoring head
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self._features: Dict[str, torch.Tensor] = {}
        self.backbone.layer4.register_forward_hook(self._save_features("layer4"))

    def _save_features(self, name: str):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        class_logits = self.defect_head(features)
        class_probs = F.softmax(class_logits, dim=1)
        anomaly_score = self.anomaly_head(features)

        return {
            "anomaly_score": anomaly_score,
            "class_logits": class_logits,
            "class_probs": class_probs,
            "features": features,
            "feature_map": self._features.get("layer4"),
        }


# ---------------------------------------------------------------------------
# Gated Attention Fusion Module
# ---------------------------------------------------------------------------

class GatedFusionModule(nn.Module):
    """Late cross-modal gated attention fusion.

    F_fused = α · F_rgb + (1 - α) · F_xray
    where α is learned via a gating network.
    """

    def __init__(self, rgb_dim: int = 2048, xray_dim: int = 512):
        super().__init__()
        combined_dim = rgb_dim + xray_dim

        # Gating network: learns α
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Projection layers to common dimension
        self.rgb_proj = nn.Linear(rgb_dim, 512)
        self.xray_proj = nn.Linear(xray_dim, 512)

        # Fused classification head
        self.fused_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_xray: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Gate value
        combined = torch.cat([f_rgb, f_xray], dim=1)
        alpha = self.gate(combined)

        # Project to common space
        rgb_proj = self.rgb_proj(f_rgb)
        xray_proj = self.xray_proj(f_xray)

        # Weighted fusion
        f_fused = alpha * rgb_proj + (1 - alpha) * xray_proj

        # Fused anomaly score
        fusion_score = self.fused_classifier(f_fused)

        return {
            "f_fused": f_fused,
            "fusion_score": fusion_score,
            "alpha": alpha,
            "rgb_proj": rgb_proj,
            "xray_proj": xray_proj,
        }


# ---------------------------------------------------------------------------
# Grad-CAM Generator
# ---------------------------------------------------------------------------

class GradCAMGenerator:
    """Generate Grad-CAM heatmaps for any CNN model.

    Usage:
        grad_cam = GradCAMGenerator(model)
        heatmap = grad_cam.generate(input_tensor, target_class=1)
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.target_layer_name = target_layer_name

    @torch.enable_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) input image tensor
            target_class: Class index to generate heatmap for.
                          If None, uses predicted class.

        Returns:
            heatmap: (H, W) numpy array normalized to [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        output = self.model(input_tensor)
        class_logits = output["class_logits"]

        if target_class is None:
            target_class = class_logits.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = class_logits[0, target_class]
        target_score.backward(retain_graph=True)

        # Get feature maps and gradients
        feature_map = self.model._features.get(self.target_layer_name)
        gradients = self.model._gradients.get(self.target_layer_name)

        if feature_map is None or gradients is None:
            # Fallback: return uniform heatmap
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        # Global average pool gradients → channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of feature maps
        cam = (weights * feature_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


# ---------------------------------------------------------------------------
# PatchCore-style Feature Bank
# ---------------------------------------------------------------------------

class PatchCoreFeatureBank:
    """Memory bank of patch-level features from normal training images.

    Implements the core idea from PatchCore (Roth et al., 2022):
    store representative patch features from known-good products and
    detect anomalies as patches far from the normal feature distribution.
    """

    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.memory_bank: Optional[torch.Tensor] = None
        self.is_fitted = False

    def fit(self, features: torch.Tensor) -> None:
        """Fit the memory bank with normal patch features.

        Args:
            features: (N, D) tensor of patch features from normal images
        """
        if features.shape[0] > self.max_samples:
            # Random subsampling (coreset selection in full PatchCore)
            indices = torch.randperm(features.shape[0])[:self.max_samples]
            features = features[indices]

        self.memory_bank = features.cpu()
        self.is_fitted = True
        logger.info("PatchCore bank fitted with %d features (dim=%d)",
                     self.memory_bank.shape[0], self.memory_bank.shape[1])

    def score(self, query_features: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores as nearest-neighbor distances.

        Args:
            query_features: (M, D) tensor of query patch features

        Returns:
            scores: (M,) tensor of anomaly scores (distances)
        """
        if not self.is_fitted or self.memory_bank is None:
            return torch.zeros(query_features.shape[0])

        # Efficient k-NN with batch distance computation
        query_features = query_features.cpu()
        distances = torch.cdist(query_features, self.memory_bank)
        min_distances, _ = distances.min(dim=1)

        return min_distances

    def save(self, path: str) -> None:
        if self.memory_bank is not None:
            torch.save({"memory_bank": self.memory_bank}, path)
            logger.info("PatchCore bank saved to %s", path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.memory_bank = data["memory_bank"]
        self.is_fitted = True
        logger.info("PatchCore bank loaded from %s (%d features)",
                     path, self.memory_bank.shape[0])


# ---------------------------------------------------------------------------
# Entropy-Based Adaptive Trigger
# ---------------------------------------------------------------------------

def compute_shannon_entropy(probs: torch.Tensor) -> float:
    """Compute Shannon entropy H(p) = -Σ p_i log(p_i).

    Higher entropy → model is more uncertain → trigger X-ray.
    """
    probs = probs.clamp(min=1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return float(entropy.item())


def normalize_entropy(entropy: float, num_classes: int) -> float:
    """Normalize entropy to [0, 1] range.

    max_entropy = log(num_classes) for uniform distribution.
    """
    max_entropy = math.log(num_classes)
    if max_entropy == 0:
        return 0.0
    return min(1.0, entropy / max_entropy)


# ---------------------------------------------------------------------------
# Unified Multimodal Inspector (orchestrates all CNN components)
# ---------------------------------------------------------------------------

class MultimodalInspector:
    """Orchestrates the full two-stage multimodal inspection pipeline.

    Manages: RGB CNN, X-ray CNN, Fusion module, Grad-CAM, PatchCore bank.
    Runs in mock mode when no trained weights are available.
    """

    def __init__(
        self,
        rgb_model: Optional[RGBAnomalyCNN] = None,
        xray_model: Optional[XRayCNN] = None,
        fusion_module: Optional[GatedFusionModule] = None,
        patch_core_bank: Optional[PatchCoreFeatureBank] = None,
        device: Optional[torch.device] = None,
        mock_mode: bool = True,
    ):
        self.device = device or DEVICE
        self.mock_mode = mock_mode

        # Initialize models
        if rgb_model is not None:
            self.rgb_model = rgb_model.to(self.device)
        else:
            self.rgb_model = RGBAnomalyCNN(pretrained=True).to(self.device)

        if xray_model is not None:
            self.xray_model = xray_model.to(self.device)
        else:
            self.xray_model = XRayCNN(pretrained=True).to(self.device)

        if fusion_module is not None:
            self.fusion_module = fusion_module.to(self.device)
        else:
            self.fusion_module = GatedFusionModule(
                rgb_dim=self.rgb_model.feature_dim,
                xray_dim=self.xray_model.feature_dim,
            ).to(self.device)

        self.patch_core_bank = patch_core_bank or PatchCoreFeatureBank()
        self.grad_cam_rgb = GradCAMGenerator(self.rgb_model, "layer4")
        self.grad_cam_xray = GradCAMGenerator(self.xray_model, "layer4")

        # Set eval mode
        self.rgb_model.eval()
        self.xray_model.eval()
        self.fusion_module.eval()

        self.model_id = "multimodal-cnn-wideresnet50-resnet34"
        self.version = "mock-v1.0" if mock_mode else "v1.0"
        self.is_ready = True

        logger.info("MultimodalInspector initialized (mock=%s, device=%s)",
                     mock_mode, self.device)

    def preprocess_rgb(self, image_np: np.ndarray) -> torch.Tensor:
        """Preprocess RGB image for CNN input.

        Args:
            image_np: (H, W, 3) uint8 numpy array

        Returns:
            tensor: (1, 3, 224, 224) normalized tensor
        """
        import cv2
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
        elif image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        image_np = cv2.resize(image_np, (224, 224))
        tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor.unsqueeze(0).to(self.device)

    def preprocess_xray(self, image_np: np.ndarray) -> torch.Tensor:
        """Preprocess grayscale X-ray image.

        Args:
            image_np: (H, W) or (H, W, 1) grayscale numpy array

        Returns:
            tensor: (1, 1, 224, 224) normalized tensor
        """
        import cv2
        if image_np.ndim == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        image_np = cv2.resize(image_np, (224, 224))
        tensor = torch.from_numpy(image_np).float().unsqueeze(0) / 255.0

        # Normalize
        tensor = (tensor - 0.449) / 0.226

        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def inspect_rgb(
        self,
        image_np: np.ndarray,
        enable_grad_cam: bool = False,
    ) -> Dict[str, Any]:
        """Run Stage 1 RGB inspection.

        Returns dict with anomaly_score, class_probs, entropy, bounding_boxes, etc.
        """
        input_tensor = self.preprocess_rgb(image_np)

        # Forward pass
        with torch.no_grad():
            output = self.rgb_model(input_tensor)

        anomaly_score = float(output["anomaly_score"].item())
        class_probs = output["class_probs"][0]
        entropy = compute_shannon_entropy(class_probs)
        norm_entropy = normalize_entropy(entropy, class_probs.shape[0])
        features = output["features"][0].cpu().numpy()

        # Class prediction
        class_idx = class_probs.argmax().item()
        class_labels = ["normal", "defective"]
        predicted_class = class_labels[min(class_idx, len(class_labels) - 1)]
        confidence = float(class_probs[class_idx].item())

        class_scores = {
            class_labels[i]: float(class_probs[i].item())
            for i in range(min(len(class_labels), class_probs.shape[0]))
        }

        # Generate bounding boxes from feature map
        bounding_boxes = self._generate_bbox_from_features(
            output.get("feature_map"), image_np.shape[:2], anomaly_score
        )

        # Grad-CAM
        grad_cam_heatmap = None
        if enable_grad_cam:
            try:
                grad_cam_heatmap = self.grad_cam_rgb.generate(
                    input_tensor.clone().detach().requires_grad_(True),
                    target_class=1,  # defective class
                )
            except Exception as e:
                logger.warning("Grad-CAM generation failed: %s", e)

        return {
            "anomaly_score": anomaly_score,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "entropy": norm_entropy,
            "class_scores": class_scores,
            "bounding_boxes": bounding_boxes,
            "features": features,
            "feature_map_shape": list(output["feature_map"].shape) if output.get("feature_map") is not None else None,
            "grad_cam_heatmap": grad_cam_heatmap,
        }

    @torch.no_grad()
    def inspect_xray(self, image_np: np.ndarray) -> Dict[str, Any]:
        """Run Stage 2 X-ray inspection."""
        input_tensor = self.preprocess_xray(image_np)

        with torch.no_grad():
            output = self.xray_model(input_tensor)

        anomaly_score = float(output["anomaly_score"].item())
        class_probs = output["class_probs"][0]
        class_idx = class_probs.argmax().item()

        class_labels = ["normal", "void", "crack"]
        predicted_class = class_labels[min(class_idx, len(class_labels) - 1)]
        confidence = float(class_probs[class_idx].item())

        class_scores = {
            class_labels[i]: float(class_probs[i].item())
            for i in range(min(len(class_labels), class_probs.shape[0]))
        }

        bounding_boxes = self._generate_bbox_from_features(
            output.get("feature_map"), image_np.shape[:2], anomaly_score
        )

        return {
            "anomaly_score": anomaly_score,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_scores": class_scores,
            "bounding_boxes": bounding_boxes,
            "features": output["features"][0].cpu().numpy(),
            "internal_defects_found": len(bounding_boxes),
        }

    @torch.no_grad()
    def fuse(
        self,
        rgb_features: np.ndarray,
        xray_features: np.ndarray,
    ) -> Dict[str, Any]:
        """Late cross-modal fusion of RGB and X-ray features."""
        f_rgb = torch.from_numpy(rgb_features).float().unsqueeze(0).to(self.device)
        f_xray = torch.from_numpy(xray_features).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.fusion_module(f_rgb, f_xray)

        return {
            "fusion_score": float(result["fusion_score"].item()),
            "alpha": float(result["alpha"].item()),
            "rgb_feature_norm": float(f_rgb.norm().item()),
            "xray_feature_norm": float(f_xray.norm().item()),
        }

    def _generate_bbox_from_features(
        self,
        feature_map: Optional[torch.Tensor],
        original_size: Tuple[int, int],
        anomaly_score: float,
    ) -> List[Dict[str, Any]]:
        """Generate bounding boxes from CNN feature activation maps."""
        if feature_map is None or anomaly_score < 0.3:
            return []

        # Use channel-wise max activation
        activation = feature_map[0].max(dim=0)[0].cpu().numpy()
        threshold = activation.mean() + activation.std()
        binary = (activation > threshold).astype(np.uint8)

        # Scale coordinates to original image size
        h_scale = original_size[0] / binary.shape[0]
        w_scale = original_size[1] / binary.shape[1]

        boxes = []
        # Simple connected component analysis
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)

        for i in range(1, min(num_features + 1, 11)):
            ys, xs = np.where(labeled == i)
            if len(ys) < 2:
                continue

            y1, y2 = int(ys.min() * h_scale), int(ys.max() * h_scale)
            x1, x2 = int(xs.min() * w_scale), int(xs.max() * w_scale)
            box_area = (x2 - x1) * (y2 - y1)
            total_area = original_size[0] * original_size[1]
            area_pct = (box_area / total_area) * 100 if total_area > 0 else 0.0

            # Severity from area percentage
            if area_pct > 25:
                severity = "Critical"
            elif area_pct > 10:
                severity = "High"
            elif area_pct > 3:
                severity = "Medium"
            else:
                severity = "Low"

            boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": "defect",
                "confidence": min(anomaly_score * 1.2, 1.0),
                "area_pct": round(area_pct, 2),
                "severity": severity,
                "defect_type": "surface_anomaly",
            })

        return sorted(boxes, key=lambda b: b["area_pct"], reverse=True)[:10]

    def load_weights(self, rgb_path: Optional[str] = None, xray_path: Optional[str] = None) -> None:
        """Load trained weights from disk."""
        if rgb_path:
            try:
                state = torch.load(rgb_path, map_location=self.device, weights_only=True)
                self.rgb_model.load_state_dict(state)
                self.mock_mode = False
                self.version = "trained-v1.0"
                logger.info("Loaded RGB weights from %s", rgb_path)
            except Exception as e:
                logger.warning("Could not load RGB weights: %s", e)

        if xray_path:
            try:
                state = torch.load(xray_path, map_location=self.device, weights_only=True)
                self.xray_model.load_state_dict(state)
                logger.info("Loaded X-ray weights from %s", xray_path)
            except Exception as e:
                logger.warning("Could not load X-ray weights: %s", e)


# ---------------------------------------------------------------------------
# Factory: create inspector with best available configuration
# ---------------------------------------------------------------------------

def create_inspector(
    rgb_weights_path: Optional[str] = None,
    xray_weights_path: Optional[str] = None,
    patch_core_path: Optional[str] = None,
) -> MultimodalInspector:
    """Create a MultimodalInspector with the best available configuration.

    Falls back to pretrained ImageNet weights (mock mode) when no
    custom weights are available.
    """
    inspector = MultimodalInspector(mock_mode=True)

    if rgb_weights_path:
        inspector.load_weights(rgb_path=rgb_weights_path)

    if xray_weights_path:
        inspector.load_weights(xray_path=xray_weights_path)

    if patch_core_path:
        try:
            inspector.patch_core_bank.load(patch_core_path)
        except Exception as e:
            logger.warning("Could not load PatchCore bank: %s", e)

    return inspector
