import logging
from pathlib import Path

import cv2
import numpy as np

from .cv_engine import CVEngine

_logger = logging.getLogger(__name__)


class SAMAdapter:
    _predictor = None
    _initialized = False
    _model_type = "vit_b"
    _checkpoint_path = None
    _device = "cpu"

    @classmethod
    def resolve_device(cls) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @classmethod
    def initialize(cls, checkpoint_path: str, device: str = "cpu") -> bool:
        if (
            cls._initialized
            and cls._predictor is not None
            and cls._checkpoint_path == checkpoint_path
            and cls._device == device
        ):
            return True

        try:
            from segment_anything import SamPredictor, sam_model_registry

            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            predictor_model = sam_model_registry[cls._model_type](checkpoint=checkpoint_path)
            predictor_model.to(device=device)
            predictor_model.eval()

            cls._predictor = SamPredictor(predictor_model)
            cls._checkpoint_path = checkpoint_path
            cls._device = device
            cls._initialized = True
            _logger.info("SAM %s loaded on %s", cls._model_type, device)
            return True
        except ImportError as exc:
            _logger.error("Missing SAM dependency: %s", exc)
            raise
        except Exception as exc:
            cls._initialized = False
            _logger.exception("Failed to initialize SAM: %s", exc)
            raise

    @classmethod
    def segment_vegetation(
        cls,
        rgb_image: np.ndarray,
        checkpoint_path: str,
        device: str = "cpu",
        min_area_ratio: float = 0.001,
        hue_center: float = 0.17,
        hue_width: float = 3.0,
    ) -> np.ndarray:
        cls.initialize(checkpoint_path, device=device)

        if cls._predictor is None:
            raise RuntimeError("SAM predictor is unavailable.")

        prompt_box = cls._find_green_region_hsv(rgb_image, hue_center=hue_center, hue_width=hue_width)
        if prompt_box is None:
            return np.zeros(rgb_image.shape[:2], dtype=np.uint8)

        try:
            import torch

            cls._predictor.set_image(rgb_image)
            with torch.inference_mode():
                masks, scores, _ = cls._predictor.predict(box=np.array(prompt_box), multimask_output=True)

            best_idx = int(np.argmax(scores))
            mask = np.where(masks[best_idx], 255, 0).astype(np.uint8)
            return cls._postprocess_mask(mask, rgb_image, min_area_ratio=min_area_ratio)
        finally:
            cls.clear_cache()

    @classmethod
    def _find_green_region_hsv(cls, rgb_image: np.ndarray, hue_center: float, hue_width: float) -> list | None:
        green_mask = CVEngine.segment_green_by_hsv(rgb_image, hue_center=hue_center, hue_width=hue_width)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(largest)
        return [int(x), int(y), int(x + width), int(y + height)]

    @classmethod
    def _postprocess_mask(cls, mask: np.ndarray, image: np.ndarray, min_area_ratio: float) -> np.ndarray:
        refined = CVEngine.refine_mask_adaptive(mask, image)
        min_area = max(24, int(mask.shape[0] * mask.shape[1] * min_area_ratio))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
        cleaned = np.zeros_like(refined)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == idx] = 255

        return cleaned

    @classmethod
    def clear_cache(cls):
        predictor = cls._predictor
        if predictor is not None:
            predictor.features = None
            predictor.is_image_set = False
            predictor.input_size = None
            predictor.original_size = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def reset(cls):
        cls._predictor = None
        cls._initialized = False
        cls._checkpoint_path = None
