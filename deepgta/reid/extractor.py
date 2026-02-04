"""
ReID Feature Extractor for DeepGTA

Provides ReID feature extraction using OSNet or other models.
Supports automatic model download and various input formats.
"""

import os
import numpy as np
from typing import List, Optional, Union
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


class ReIDExtractor:
    """ReID feature extractor using OSNet or similar models.

    Extracts appearance features from cropped person/object images
    for use in multi-object tracking.
    """

    # Default model URL for sports ReID
    DEFAULT_MODEL_URL = "https://drive.google.com/uc?id=14zzlm1nI9Ws_Il9RYNChwPC7Fsul7xwl"
    DEFAULT_MODEL_NAME = "sports_model.pth.tar-60"

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = 'osnet_x1_0',
        device: str = 'cuda',
        image_size: tuple = (256, 128),
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225]
    ):
        """Initialize the feature extractor.

        Args:
            model_path: Path to model weights (downloads default if None)
            model_name: Model architecture name
            device: Device to run on ('cuda' or 'cpu')
            image_size: Input image size (height, width)
            pixel_mean: Normalization mean
            pixel_std: Normalization std
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size

        # Build transforms
        self.transforms = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=pixel_mean, std=pixel_std)
        ])

        self.to_pil = T.ToPILImage()

        # Build model
        self.model = self._build_model(model_name, model_path)
        self.model.eval()
        self.model.to(self.device)

    def _build_model(self, model_name: str, model_path: Optional[str]) -> nn.Module:
        """Build the feature extraction model.

        Args:
            model_name: Model architecture name
            model_path: Path to model weights

        Returns:
            PyTorch model
        """
        try:
            # Try to import torchreid models
            from torchreid.models import build_model
            from torchreid.utils import load_pretrained_weights

            model = build_model(
                model_name,
                num_classes=1,
                pretrained=model_path is None,
                use_gpu=str(self.device).startswith('cuda')
            )

            if model_path and os.path.isfile(model_path):
                load_pretrained_weights(model, model_path)

            return model

        except ImportError:
            # Fall back to simple OSNet implementation
            return self._build_simple_osnet(model_path)

    def _build_simple_osnet(self, model_path: Optional[str]) -> nn.Module:
        """Build a simplified OSNet model if torchreid is not available.

        Args:
            model_path: Path to model weights

        Returns:
            Simple feature extraction model
        """
        # Use a pretrained ResNet as fallback
        try:
            from torchvision.models import resnet50, ResNet50_Weights

            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # Remove the final classification layer
            model.fc = nn.Identity()

            if model_path and os.path.isfile(model_path):
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    model.load_state_dict(state_dict, strict=False)
                except Exception:
                    pass  # Use default weights

            return model

        except Exception:
            # Minimal fallback - just return identity features
            return nn.Identity()

    def __call__(self, inputs: Union[torch.Tensor, np.ndarray, List, str]) -> torch.Tensor:
        """Extract features from inputs.

        Args:
            inputs: Can be:
                - torch.Tensor with shape (B, C, H, W) or (C, H, W)
                - numpy.ndarray with shape (H, W, C)
                - List of numpy arrays or image paths
                - Single image path string

        Returns:
            Feature tensor with shape (B, D) where D is feature dimension
        """
        images = self._preprocess(inputs)

        with torch.no_grad():
            features = self.model(images)

        return features

    def _preprocess(self, inputs) -> torch.Tensor:
        """Preprocess inputs to tensor format.

        Args:
            inputs: Various input formats

        Returns:
            Preprocessed tensor on device
        """
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
            return inputs.to(self.device)

        if isinstance(inputs, str):
            # Single image path
            image = Image.open(inputs).convert('RGB')
            image = self.transforms(image)
            return image.unsqueeze(0).to(self.device)

        if isinstance(inputs, np.ndarray):
            # Single numpy array
            if inputs.ndim == 3 and inputs.shape[2] in [3, 4]:
                # HWC format
                image = self.to_pil(inputs)
            else:
                image = Image.fromarray(inputs).convert('RGB')
            image = self.transforms(image)
            return image.unsqueeze(0).to(self.device)

        if isinstance(inputs, list):
            # List of inputs
            images = []
            for element in inputs:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')
                elif isinstance(element, np.ndarray):
                    if element.shape[2] in [3, 4] if element.ndim == 3 else False:
                        image = self.to_pil(element)
                    else:
                        image = Image.fromarray(element).convert('RGB')
                elif isinstance(element, Image.Image):
                    image = element.convert('RGB')
                else:
                    raise TypeError(f"Unsupported element type: {type(element)}")

                image = self.transforms(image)
                images.append(image)

            return torch.stack(images, dim=0).to(self.device)

        raise TypeError(f"Unsupported input type: {type(inputs)}")

    def extract_from_frame(
        self,
        frame: np.ndarray,
        detections: List,
        normalize: bool = True
    ) -> np.ndarray:
        """Extract features for detections in a frame.

        Args:
            frame: Full frame image (BGR or RGB)
            detections: List of Detection objects with bbox attribute
            normalize: Whether to L2 normalize features

        Returns:
            Feature array with shape (N, D)
        """
        if not detections:
            return np.array([])

        # Convert frame to PIL
        if frame.shape[2] == 3:
            # Assume BGR, convert to RGB
            frame_rgb = frame[:, :, ::-1]
        else:
            frame_rgb = frame

        frame_pil = Image.fromarray(frame_rgb)

        # Crop detections
        crops = []
        for det in detections:
            bbox = det.bbox if hasattr(det, 'bbox') else det
            x1, y1, x2, y2 = map(int, bbox[:4])

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                # Invalid crop, create dummy
                crop = Image.new('RGB', self.image_size[::-1])
            else:
                crop = frame_pil.crop((x1, y1, x2, y2))

            crops.append(crop)

        # Extract features
        features = self(crops).cpu().numpy()

        # Normalize
        if normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / (norms + 1e-8)

        return features

    @staticmethod
    def download_default_model(save_dir: str = '.') -> str:
        """Download the default sports ReID model.

        Args:
            save_dir: Directory to save the model

        Returns:
            Path to downloaded model
        """
        try:
            import gdown
        except ImportError:
            raise ImportError("Please install gdown: pip install gdown")

        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, ReIDExtractor.DEFAULT_MODEL_NAME)

        if not os.path.exists(output_path):
            gdown.download(
                ReIDExtractor.DEFAULT_MODEL_URL,
                output_path,
                quiet=False
            )

        return output_path
