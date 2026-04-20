import base64
import gc
import io
import logging
import os
import time

import cv2
import numpy as np
from PIL import Image

from odoo import fields, models

from .cv_engine import CVEngine

_logger = logging.getLogger(__name__)


class LAICalculation(models.Model):
    _name = "lai.calculation"
    _description = "LAI Calculation Record"

    name = fields.Char(string="Name", required=True)
    image = fields.Binary(string="Original Image", attachment=True)
    image_filename = fields.Char(string="Filename")

    crop_type = fields.Selection(
        [
            ("wheat", "Озимая пшеница"),
            ("corn", "Кукуруза"),
            ("sunflower", "Подсолнечник"),
            ("soy", "Соя"),
            ("mixed", "Смешанная/Отдельные листья"),
            ("vertical", "Вертикальная поверхность"),
        ],
        string="Культура",
        default="mixed",
    )

    lai_avg = fields.Float(string="Средний LAI", digits=(4, 2))
    lai_heatmap = fields.Binary(string="Heatmap", attachment=True)
    lai_heatmap_filename = fields.Char(string="Heatmap Filename")

    confidence = fields.Float(string="Уверенность модели", digits=(4, 3), default=0.0)
    coverage_percent = fields.Float(string="Покрытие растительностью %", digits=(4, 2))
    texture_homogeneity = fields.Float(string="Текстура (однородность)", digits=(4, 3))
    recommendation_text = fields.Text(string="Рекомендация системы")

    segmentation_method = fields.Selection(
        [
            ("cv_ensemble", "CV Ensemble (OpenCV + Indices)"),
            ("sam", "Segment Anything Model (SAM vit_b)"),
        ],
        string="Метод сегментации",
        default="cv_ensemble",
    )

    crop_status = fields.Char(string="AI Статус посевов", readonly=True)
    status_confidence = fields.Float(string="Уверенность AI", digits=(4, 3), readonly=True)

    user_id = fields.Many2one("res.users", string="User", default=lambda self: self.env.user)
    date_calculated = fields.Datetime(string="Calculated On", default=fields.Datetime.now)

    use_custom_calibration = fields.Boolean(string="Custom Calibration", default=False)
    custom_green_hue_center = fields.Float(string="Green Hue Center", default=0.17)
    custom_green_hue_width = fields.Float(string="Hue Sensitivity", default=3.0)
    custom_lai_min = fields.Float(string="Min LAI", default=0.5)
    custom_lai_max = fields.Float(string="Max LAI", default=6.0)

    def check_access_rights(self, operation, raise_exception=True):
        if self.env.user._is_public() and operation == "read":
            return True
        return super().check_access_rights(operation, raise_exception)

    def _process_image_and_calculate_lai(self, image_data: bytes, crop_type: str):
        img_arr = proc_image = work_image = mask = score_map = None
        indices = lai_result = confidence_data = status_info = None

        try:
            img_arr = self._load_image(image_data)
            if img_arr is None or img_arr.size == 0:
                raise ValueError("Failed to load or decode image.")

            proc_image, work_image = self._prepare_working_image(img_arr)
            mask, indices, score_map = self._segment_image(proc_image)

            lai_result = CVEngine.calculate_lai_from_mask(
                mask,
                proc_image,
                crop_type=crop_type,
                lai_min=self.custom_lai_min if self.use_custom_calibration else 0.0,
                lai_max=self.custom_lai_max if self.use_custom_calibration else 6.0,
                indices=indices,
                vegetation_score=score_map,
            )
            confidence_data = CVEngine.estimate_confidence_v2(proc_image, mask, lai_result)
            heatmap_bytes = self._generate_heatmap_pil(
                proc_image,
                mask,
                lai_result["lai_value"],
                confidence_data["confidence"],
                score_map=score_map,
            )

            status_info = self._assess_crop_status_with_clip(work_image if work_image is not None else proc_image)
            recommendation = self._build_recommendation(confidence_data, status_info)

            avg_lai = float(lai_result["lai_value"])
            coverage = float(lai_result["coverage"] * 100.0)
            texture_hom = float(lai_result["texture_homogeneity"])
            confidence = float(confidence_data["confidence"])

            _logger.info(
                "LAI calculated: method=%s, lai=%.2f, coverage=%.1f%%, conf=%.3f",
                self.segmentation_method or "cv_ensemble",
                avg_lai,
                coverage,
                confidence,
            )

            return (
                avg_lai,
                heatmap_bytes,
                "lai_heatmap.jpg",
                confidence,
                coverage,
                texture_hom,
                recommendation,
                status_info,
            )
        except ValueError:
            raise
        except Exception as exc:
            _logger.exception("Unexpected error in LAI calculation")
            raise ValueError(f"LAI calculation failed: {exc}") from exc
        finally:
            self._cleanup_memory([img_arr, proc_image, work_image, mask, indices, score_map, lai_result, confidence_data, status_info])

    def _load_image(self, image_data: bytes) -> np.ndarray | None:
        np_buffer = np.frombuffer(image_data, dtype=np.uint8)
        decoded = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if decoded is not None:
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            return self._normalize_image_size(rgb)

        try:
            with io.BytesIO(image_data) as buffer:
                with Image.open(buffer) as pil_img:
                    pil_img = pil_img.convert("RGB")
                    return self._normalize_image_size(np.array(pil_img))
        except Exception as exc:
            _logger.error("Image decoding failed: %s", exc)
            return None

    def _normalize_image_size(self, image: np.ndarray) -> np.ndarray:
        normalized, _ = CVEngine.downscale_for_processing(image, max_dim=1600)
        return normalized

    def _prepare_working_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        working_image, _ = CVEngine.downscale_for_processing(image, max_dim=CVEngine.WORKING_MAX_DIM)
        processed = CVEngine.preprocess_image(working_image)
        return processed, working_image

    def _segment_image(self, image: np.ndarray) -> tuple[np.ndarray, dict | None, np.ndarray | None]:
        method = self.segmentation_method or "cv_ensemble"
        if method == "sam":
            mask = self._segment_with_sam_pipeline(image)
            if mask is not None:
                return mask, None, None

        return self._segment_with_cv_pipeline(image)

    def _segment_with_sam_pipeline(self, image: np.ndarray) -> np.ndarray | None:
        from .sam_adapter import SAMAdapter

        start_time = time.time()
        checkpoint_path = os.environ.get(
            "SAM_CHECKPOINT_PATH",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "sam_checkpoints", "sam_vit_b_01ec64.pth"),
        )

        try:
            mask = SAMAdapter.segment_vegetation(
                image,
                checkpoint_path=checkpoint_path,
                device=SAMAdapter.resolve_device(),
                min_area_ratio=0.0015,
                hue_center=self.custom_green_hue_center if self.use_custom_calibration else 0.17,
                hue_width=self.custom_green_hue_width if self.use_custom_calibration else 3.0,
            )
            elapsed = time.time() - start_time
            coverage = float(np.count_nonzero(mask) / mask.size) * 100.0
            _logger.info("[SAM vit_b] Done in %.2fs | coverage: %.1f%%", elapsed, coverage)
            return mask
        except Exception as exc:
            _logger.warning("[SAM] Failed: %s. Falling back to CV ensemble.", exc)
            return None

    def _segment_with_cv_pipeline(self, image: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
        start_time = time.time()
        indices = CVEngine.calculate_vegetation_indices(image)
        score_map = CVEngine.build_vegetation_score(indices, image, crop_type=self.crop_type or "mixed")
        mask = CVEngine.ensemble_segmentation(
            indices,
            image,
            hue_center=self.custom_green_hue_center if self.use_custom_calibration else 0.17,
            hue_width=self.custom_green_hue_width if self.use_custom_calibration else 3.0,
            crop_type=self.crop_type or "mixed",
            vegetation_score=score_map,
        )
        elapsed = time.time() - start_time
        coverage = float(np.count_nonzero(mask) / mask.size) * 100.0
        _logger.info("[CV Ensemble] Done in %.2fs | coverage: %.1f%%", elapsed, coverage)
        return mask, indices, score_map

    def _generate_heatmap_pil(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        lai_value: float,
        confidence: float,
        score_map: np.ndarray | None = None,
    ) -> bytes:
        if score_map is None:
            indices = CVEngine.calculate_vegetation_indices(image)
            score_map = CVEngine.build_vegetation_score(indices, image, crop_type=self.crop_type or "mixed")
        score_uint8 = np.clip(score_map * 255.0, 0, 255).astype(np.uint8)
        colored_map = cv2.applyColorMap(score_uint8, cv2.COLORMAP_TURBO)
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

        background = (image.astype(np.float32) * 0.38).astype(np.uint8)
        alpha = np.clip((score_map ** 1.15)[..., None] * 0.82, 0.0, 0.82)
        heatmap = ((colored_map.astype(np.float32) * alpha) + (background.astype(np.float32) * (1.0 - alpha))).astype(np.uint8)

        visible_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(heatmap, contours, -1, (240, 255, 240), 1)

        cv2.putText(
            heatmap,
            f"LAI: {lai_value:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            heatmap,
            f"Confidence: {confidence:.2f}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        result_img = Image.fromarray(heatmap)
        buffer = io.BytesIO()
        result_img.save(buffer, format="JPEG", quality=82, optimize=True)
        return buffer.getvalue()

    def _assess_crop_status_with_clip(self, rgb_image: np.ndarray) -> dict:
        from .clip_adapter import CLIPAdapter

        if not CLIPAdapter.is_available():
            return {"status": "unknown", "confidence": 0.0, "all_probabilities": {}}

        try:
            return CLIPAdapter.assess_crop_status(rgb_image)
        except Exception as exc:
            _logger.warning("CLIP assessment skipped: %s", exc)
            return {"status": "unknown", "confidence": 0.0, "all_probabilities": {}}

    def _build_recommendation(self, confidence_data: dict, status_info: dict) -> str:
        recommendation = confidence_data["recommendation"]
        status = status_info.get("status", "unknown")
        status_confidence = float(status_info.get("confidence", 0.0))

        if status != "unknown" and status_confidence >= 0.55:
            recommendation += f"\n\nAI assessment: {status.title()} ({status_confidence * 100:.0f}%)"

            status_lower = status.lower()
            if any(keyword in status_lower for keyword in ("stressed", "yellowing", "drought", "deficiency")):
                recommendation += "\nPossible crop stress detected. Check water supply, soil moisture and nutrient status."
            elif any(keyword in status_lower for keyword in ("weed", "infested")):
                recommendation += "\nMixed vegetation is likely present. Inspect the area for weed pressure."
            elif any(keyword in status_lower for keyword in ("bare", "sparse", "early")):
                recommendation += "\nVegetation density is low. Re-shoot a denser area if you need canopy-level LAI."

        return recommendation

    def _cleanup_memory(self, variables: list):
        for var in variables:
            if var is not None:
                del var
        gc.collect()
