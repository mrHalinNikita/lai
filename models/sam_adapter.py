import gc
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
    _sam_input_max_dim = 768

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

        cls.reset()

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
        crop_type: str = "mixed",
    ) -> np.ndarray:
        cls.initialize(checkpoint_path, device=device)

        if cls._predictor is None:
            raise RuntimeError("SAM predictor is unavailable.")

        scaled_image, scale = CVEngine.downscale_for_processing(rgb_image, max_dim=cls._sam_input_max_dim)
        support_data = cls._build_support_prior(
            scaled_image,
            hue_center=hue_center,
            hue_width=hue_width,
            crop_type=crop_type,
        )
        prompt_box = cls._find_prompt_box(support_data["candidate_mask"])
        if prompt_box is None:
            return np.zeros(rgb_image.shape[:2], dtype=np.uint8)

        try:
            import torch

            cls._predictor.set_image(scaled_image)
            with torch.inference_mode():
                masks, _, _ = cls._predictor.predict(box=np.array(prompt_box), multimask_output=False)

            sam_mask = np.where(masks[0], 255, 0).astype(np.uint8)
            fused_mask = cls._fuse_with_support(
                sam_mask,
                support_data["support_mask"],
                scaled_image,
                min_area_ratio=min_area_ratio,
                crop_type=crop_type,
            )

            if scale != 1.0:
                fused_mask = cv2.resize(
                    fused_mask,
                    (rgb_image.shape[1], rgb_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            return fused_mask
        finally:
            cls.reset()

    @classmethod
    def _build_support_prior(
        cls,
        rgb_image: np.ndarray,
        hue_center: float,
        hue_width: float,
        crop_type: str,
    ) -> dict:
        indices = CVEngine.calculate_vegetation_indices(rgb_image)
        vegetation_score = CVEngine.build_vegetation_score(indices, rgb_image, crop_type=crop_type)
        support = CVEngine.build_support_masks(indices, rgb_image, hue_center=hue_center, hue_width=hue_width)

        candidate_mask = support["broad_support"] | support["strong_support"]
        candidate_scores = vegetation_score[candidate_mask]
        if candidate_scores.size:
            score_threshold = max(0.16, float(np.percentile(candidate_scores, 52)))
        else:
            score_threshold = 0.18

        support_mask = (
            support["strong_support"]
            | ((vegetation_score >= score_threshold) & support["broad_support"])
            | ((vegetation_score >= max(0.12, score_threshold - 0.05)) & support["hsv_mask"])
        )
        support_mask = np.where(support_mask, 255, 0).astype(np.uint8)
        support_mask = cv2.morphologyEx(
            support_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        return {
            "candidate_mask": support_mask,
            "support_mask": support_mask,
        }

    @classmethod
    def _find_prompt_box(cls, candidate_mask: np.ndarray) -> list | None:
        ys, xs = np.where(candidate_mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None

        x1 = int(np.percentile(xs, 2))
        x2 = int(np.percentile(xs, 98))
        y1 = int(np.percentile(ys, 2))
        y2 = int(np.percentile(ys, 98))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @classmethod
    def _fuse_with_support(
        cls,
        sam_mask: np.ndarray,
        support_mask: np.ndarray,
        image: np.ndarray,
        min_area_ratio: float,
        crop_type: str,
    ) -> np.ndarray:
        support_bool = support_mask > 0
        sam_bool = sam_mask > 0

        overlap_pixels = np.count_nonzero(sam_bool & support_bool)
        sam_pixels = max(np.count_nonzero(sam_bool), 1)
        overlap_ratio = float(overlap_pixels / sam_pixels)

        if overlap_ratio < 0.12:
            fused = support_mask.copy()
        else:
            dilated_support = cv2.dilate(support_mask, np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
            dilated_sam = cv2.dilate(sam_mask, np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
            fused_bool = (sam_bool & dilated_support) | (support_bool & dilated_sam)
            fused = np.where(fused_bool, 255, 0).astype(np.uint8)

        refined = CVEngine.refine_mask_adaptive(fused, image, crop_type=crop_type)
        min_area = max(24, int(image.shape[0] * image.shape[1] * min_area_ratio))

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
            try:
                predictor.features = None
                predictor.is_image_set = False
                predictor.input_size = None
                predictor.original_size = None
            except Exception:
                pass

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def reset(cls):
        predictor = cls._predictor
        cls._predictor = None
        cls._initialized = False
        cls._checkpoint_path = None

        if predictor is not None:
            try:
                model = getattr(predictor, "model", None)
                if model is not None:
                    try:
                        model.cpu()
                    except Exception:
                        pass
                    del model
                del predictor
            except Exception:
                pass

        cls.clear_cache()
        gc.collect()
