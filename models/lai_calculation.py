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

    name = fields.Char(string="Название", required=True)
    image = fields.Binary(string="Исходное изображение", attachment=True)
    image_filename = fields.Char(string="Имя файла")

    crop_type = fields.Selection(
        [
            ("wheat", "Озимая пшеница"),
            ("corn", "Кукуруза"),
            ("sunflower", "Подсолнечник"),
            ("soy", "Соя"),
            ("mixed", "Смешанная / универсальная сцена"),
            ("vertical", "Вертикальная поверхность"),
        ],
        string="Тип сцены",
        default="mixed",
    )

    lai_avg = fields.Float(string="Средний LAI", digits=(4, 2))
    lai_heatmap = fields.Binary(string="Сводная тепловая карта", attachment=True)
    lai_heatmap_filename = fields.Char(string="Имя файла карты")
    vegetation_map = fields.Binary(string="Карта растительности", attachment=True)
    vegetation_map_filename = fields.Char(string="Имя файла карты растительности")
    lai_distribution_map = fields.Binary(string="Карта распределения LAI", attachment=True)
    lai_distribution_map_filename = fields.Char(string="Имя файла карты LAI")

    confidence = fields.Float(string="Уверенность модели", digits=(4, 3), default=0.0)
    coverage_percent = fields.Float(string="Покрытие растительностью %", digits=(4, 2))
    green_fraction_percent = fields.Float(string="Зеленая доля %", digits=(4, 2))
    gap_fraction_percent = fields.Float(string="Доля просветов %", digits=(4, 2))
    effective_k = fields.Float(string="Эффективный k", digits=(4, 3))
    texture_homogeneity = fields.Float(string="Однородность текстуры", digits=(4, 3))
    lai_stddev = fields.Float(string="Стандартное отклонение LAI", digits=(4, 3))
    lai_p10 = fields.Float(string="LAI P10", digits=(4, 3))
    lai_p90 = fields.Float(string="LAI P90", digits=(4, 3))
    illumination_class = fields.Char(string="Тип освещения")
    method_version = fields.Char(string="Версия метода")
    recommendation_text = fields.Text(string="Рекомендация системы")

    segmentation_method = fields.Selection(
        [
            ("cv_ensemble", "Научный RGB (Beer-Lambert + индексы)"),
            ("sam", "Segment Anything Model (экспериментально)"),
        ],
        string="Режим сегментации",
        default="cv_ensemble",
    )

    crop_status = fields.Char(string="AI-статус", readonly=True)
    status_confidence = fields.Float(string="Уверенность AI", digits=(4, 3), readonly=True)

    user_id = fields.Many2one("res.users", string="Пользователь", default=lambda self: self.env.user)
    date_calculated = fields.Datetime(string="Дата расчета", default=fields.Datetime.now)

    use_custom_calibration = fields.Boolean(string="Пользовательская калибровка", default=False)
    custom_green_hue_center = fields.Float(string="Центр зеленого оттенка", default=0.17)
    custom_green_hue_width = fields.Float(string="Чувствительность оттенка", default=3.0)
    custom_lai_min = fields.Float(string="Минимальный LAI", default=0.0)
    custom_lai_max = fields.Float(string="Максимальный LAI", default=6.0)

    def check_access_rights(self, operation, raise_exception=True):
        if self.env.user._is_public() and operation == "read":
            return True
        return super().check_access_rights(operation, raise_exception)

    def _process_image_and_calculate_lai(self, image_data: bytes, crop_type: str):
        img_arr = proc_image = work_image = analysis_image = None
        mask = score_map = density_map = local_lai_map = display_map = None
        indices = lai_result = confidence_data = status_info = None
        heatmap_bytes = vegetation_map_bytes = lai_distribution_map_bytes = None
        analysis_bbox = None

        try:
            img_arr = self._load_image(image_data)
            if img_arr is None or img_arr.size == 0:
                raise ValueError("Не удалось загрузить или декодировать изображение.")

            proc_image, work_image = self._prepare_working_image(img_arr)
            analysis_image, analysis_bbox = self._extract_analysis_region(proc_image)
            mask, indices, score_map = self._segment_image(analysis_image)

            lai_result = CVEngine.calculate_lai_from_mask(
                mask,
                analysis_image,
                crop_type=crop_type,
                lai_min=self.custom_lai_min if self.use_custom_calibration else 0.0,
                lai_max=self.custom_lai_max if self.use_custom_calibration else 6.0,
                indices=indices,
                vegetation_score=score_map,
            )
            density_map = lai_result.get("density_map")
            local_lai_map = lai_result.get("local_lai_map")
            display_map = self._build_display_map(score_map, density_map, local_lai_map)
            confidence_data = CVEngine.estimate_confidence_v2(analysis_image, mask, lai_result)

            vegetation_map_bytes = self._generate_vegetation_map_pil(
                proc_image,
                analysis_image,
                mask,
                score_map=score_map,
                density_map=density_map,
                analysis_bbox=analysis_bbox,
            )
            lai_distribution_map_bytes = self._generate_lai_map_pil(
                proc_image,
                analysis_image,
                mask,
                local_lai_map,
                lai_result["lai_value"],
                confidence_data["confidence"],
                analysis_bbox=analysis_bbox,
            )
            heatmap_bytes = self._generate_heatmap_pil(
                proc_image,
                analysis_image,
                mask,
                lai_result["lai_value"],
                confidence_data["confidence"],
                score_map=display_map,
                analysis_bbox=analysis_bbox,
            )

            best_analysis_image = analysis_image if analysis_image is not None else (work_image if work_image is not None else proc_image)
            status_info = self._assess_crop_status_with_clip(best_analysis_image)
            recommendation = self._build_recommendation(confidence_data, status_info, lai_result, analysis_bbox)

            result = {
                "avg_lai": float(lai_result["lai_value"]),
                "heatmap_bytes": heatmap_bytes,
                "heatmap_filename": "lai_heatmap.jpg",
                "vegetation_map_bytes": vegetation_map_bytes,
                "vegetation_map_filename": "vegetation_map.jpg",
                "lai_distribution_map_bytes": lai_distribution_map_bytes,
                "lai_distribution_map_filename": "lai_distribution_map.jpg",
                "confidence": float(confidence_data["confidence"]),
                "coverage_percent": float(lai_result["apparent_cover"] * 100.0),
                "green_fraction_percent": float(lai_result["green_fraction"] * 100.0),
                "gap_fraction_percent": float(lai_result["gap_fraction"] * 100.0),
                "effective_k": float(lai_result["extinction_coefficient"]),
                "texture_homogeneity": float(lai_result["texture_homogeneity"]),
                "lai_stddev": float(lai_result.get("local_lai_std", 0.0)),
                "lai_p10": float(lai_result.get("local_lai_p10", 0.0)),
                "lai_p90": float(lai_result.get("local_lai_p90", 0.0)),
                "illumination_class": lai_result.get("illumination_class", "неизвестно"),
                "method_version": lai_result.get("method_version", CVEngine.SCIENTIFIC_METHOD),
                "recommendation": recommendation,
                "status_info": status_info,
            }

            _logger.info(
                "LAI calculated: method=%s, scene=%s, lai=%.2f, cover=%.1f%%, gap=%.1f%%, conf=%.3f",
                result["method_version"],
                crop_type,
                result["avg_lai"],
                result["coverage_percent"],
                result["gap_fraction_percent"],
                result["confidence"],
            )
            return result
        except ValueError:
            raise
        except Exception as exc:
            _logger.exception("Unexpected error in LAI calculation")
            raise ValueError(f"LAI calculation failed: {exc}") from exc
        finally:
            self._cleanup_memory(
                [
                    img_arr,
                    proc_image,
                    work_image,
                    analysis_image,
                    mask,
                    indices,
                    score_map,
                    density_map,
                    local_lai_map,
                    display_map,
                    lai_result,
                    confidence_data,
                    status_info,
                    heatmap_bytes,
                    vegetation_map_bytes,
                    lai_distribution_map_bytes,
                ]
            )

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

    def _extract_analysis_region(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        bbox = CVEngine.detect_analysis_roi(image)
        if bbox is None:
            return image, None

        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return image, None
        return roi, bbox

    def _segment_image(self, image: np.ndarray) -> tuple[np.ndarray, dict | None, np.ndarray | None]:
        method = self.segmentation_method or "cv_ensemble"
        if method == "sam":
            mask = self._segment_with_sam_pipeline(image)
            if mask is not None:
                indices = CVEngine.calculate_vegetation_indices(image)
                score_map = CVEngine.build_vegetation_score(indices, image, crop_type=self.crop_type or "mixed")
                return mask, indices, score_map

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
                crop_type=self.crop_type or "mixed",
            )
            elapsed = time.time() - start_time
            coverage = float(np.count_nonzero(mask) / mask.size) * 100.0
            _logger.info("[SAM] Done in %.2fs | coverage: %.1f%%", elapsed, coverage)
            return mask
        except Exception as exc:
            _logger.warning("[SAM] Failed: %s. Falling back to scientific RGB pipeline.", exc)
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
        _logger.info("[Scientific RGB] Done in %.2fs | coverage: %.1f%%", elapsed, coverage)
        return mask, indices, score_map

    def _build_display_map(
        self,
        score_map: np.ndarray | None,
        density_map: np.ndarray | None,
        local_lai_map: np.ndarray | None,
    ) -> np.ndarray | None:
        if score_map is None and density_map is None and local_lai_map is None:
            return None

        if score_map is not None and density_map is not None:
            display_map = (0.64 * score_map.astype(np.float32)) + (0.36 * density_map.astype(np.float32))
        elif score_map is not None:
            display_map = score_map.astype(np.float32)
        else:
            display_map = density_map.astype(np.float32)

        if local_lai_map is not None:
            lai_cap = max(self.custom_lai_max if self.use_custom_calibration else 6.0, 0.25)
            normalized_lai = np.clip(local_lai_map.astype(np.float32) / lai_cap, 0.0, 1.0)
            display_map = (0.82 * display_map) + (0.18 * normalized_lai)

        return np.clip(display_map, 0.0, 1.0).astype(np.float32)

    def _generate_heatmap_pil(
        self,
        full_image: np.ndarray,
        analysis_image: np.ndarray,
        mask: np.ndarray,
        lai_value: float,
        confidence: float,
        score_map: np.ndarray | None = None,
        analysis_bbox: tuple[int, int, int, int] | None = None,
    ) -> bytes:
        if score_map is None:
            indices = CVEngine.calculate_vegetation_indices(analysis_image)
            base_score = CVEngine.build_vegetation_score(indices, analysis_image, crop_type=self.crop_type or "mixed")
            score_map = CVEngine.build_density_map(mask, base_score)

        score_blurred = cv2.GaussianBlur(score_map.astype(np.float32), (0, 0), sigmaX=2.4, sigmaY=2.4)
        score_norm = CVEngine.normalize_map(score_blurred, 12.0, 98.0)
        score_uint8 = np.clip(score_norm * 255.0, 0, 255).astype(np.uint8)
        colored_map = cv2.applyColorMap(score_uint8, cv2.COLORMAP_TURBO)
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

        roi_background = ((0.88 * analysis_image.astype(np.float32)) + 6.0).clip(0, 255).astype(np.uint8)
        mask_float = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
        support_alpha = np.clip((0.20 + (score_norm ** 1.02) * 0.50), 0.20, 0.72)
        support_alpha = np.clip(support_alpha + (mask_float * 0.06), 0.20, 0.78)[..., None]
        roi_overlay = ((colored_map.astype(np.float32) * support_alpha) + (roi_background.astype(np.float32) * (1.0 - support_alpha))).astype(np.uint8)

        visible_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_overlay, contours, -1, (248, 250, 236), 1)

        heatmap = ((0.96 * full_image.astype(np.float32)) + 4.0).clip(0, 255).astype(np.uint8)

        if analysis_bbox is None:
            heatmap = roi_overlay
        else:
            x1, y1, x2, y2 = analysis_bbox
            heatmap[y1:y2, x1:x2] = roi_overlay
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (244, 248, 232), 2)

        legend_h = 12
        legend_w = max(120, min(220, full_image.shape[1] // 3))
        legend_x = 14
        legend_y = full_image.shape[0] - 28
        legend = np.tile(np.linspace(0, 255, legend_w, dtype=np.uint8), (legend_h, 1))
        legend = cv2.applyColorMap(legend, cv2.COLORMAP_TURBO)
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        heatmap[legend_y : legend_y + legend_h, legend_x : legend_x + legend_w] = legend
        cv2.rectangle(heatmap, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (255, 255, 255), 1)
        cv2.putText(heatmap, "Low", (legend_x, legend_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(heatmap, "High", (legend_x + legend_w - 30, legend_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        cv2.putText(heatmap, f"LAI: {lai_value:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
        cv2.putText(heatmap, f"Conf: {confidence:.2f}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

        result_img = Image.fromarray(heatmap)
        buffer = io.BytesIO()
        result_img.save(buffer, format="JPEG", quality=86, optimize=True)
        return buffer.getvalue()

    def _generate_vegetation_map_pil(
        self,
        full_image: np.ndarray,
        analysis_image: np.ndarray,
        mask: np.ndarray,
        score_map: np.ndarray | None = None,
        density_map: np.ndarray | None = None,
        analysis_bbox: tuple[int, int, int, int] | None = None,
    ) -> bytes:
        if score_map is None:
            indices = CVEngine.calculate_vegetation_indices(analysis_image)
            score_map = CVEngine.build_vegetation_score(indices, analysis_image, crop_type=self.crop_type or "mixed")
        if density_map is None:
            density_map = CVEngine.build_density_map(mask, score_map)

        merged = (0.55 * score_map.astype(np.float32)) + (0.45 * density_map.astype(np.float32))
        score_norm = CVEngine.normalize_map(merged, 4.0, 98.0)
        score_uint8 = np.clip(score_norm * 255.0, 0, 255).astype(np.uint8)
        colored_map = cv2.applyColorMap(score_uint8, cv2.COLORMAP_SUMMER)
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

        roi_background = ((0.72 * analysis_image.astype(np.float32)) + 22.0).clip(0, 255).astype(np.uint8)
        alpha = np.clip(0.18 + score_norm[..., None] * 0.60, 0.18, 0.74)
        roi_overlay = ((colored_map.astype(np.float32) * alpha) + (roi_background.astype(np.float32) * (1.0 - alpha))).astype(np.uint8)
        contours, _ = cv2.findContours(np.where(mask > 0, 255, 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_overlay, contours, -1, (245, 247, 236), 1)

        output = full_image.copy()
        if analysis_bbox is None:
            output = roi_overlay
        else:
            x1, y1, x2, y2 = analysis_bbox
            output[y1:y2, x1:x2] = roi_overlay
            cv2.rectangle(output, (x1, y1), (x2, y2), (244, 248, 232), 2)

        cv2.putText(output, "Vegetation map", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(output, "Low support -> High support", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        result_img = Image.fromarray(output)
        buffer = io.BytesIO()
        result_img.save(buffer, format="JPEG", quality=86, optimize=True)
        return buffer.getvalue()

    def _generate_lai_map_pil(
        self,
        full_image: np.ndarray,
        analysis_image: np.ndarray,
        mask: np.ndarray,
        local_lai_map: np.ndarray | None,
        lai_value: float,
        confidence: float,
        analysis_bbox: tuple[int, int, int, int] | None = None,
    ) -> bytes:
        if local_lai_map is None:
            local_lai_map = np.zeros(analysis_image.shape[:2], dtype=np.float32)

        lai_cap = max(self.custom_lai_max if self.use_custom_calibration else 6.0, lai_value, 0.25)
        normalized_lai = np.clip(local_lai_map.astype(np.float32) / lai_cap, 0.0, 1.0)
        lai_uint8 = np.clip(normalized_lai * 255.0, 0, 255).astype(np.uint8)
        colored_map = cv2.applyColorMap(lai_uint8, cv2.COLORMAP_TURBO)
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

        roi_background = ((0.82 * analysis_image.astype(np.float32)) + 10.0).clip(0, 255).astype(np.uint8)
        alpha = np.clip(0.16 + normalized_lai[..., None] * 0.64, 0.16, 0.78)
        alpha = np.where(mask[..., None] > 0, alpha, 0.08)
        roi_overlay = ((colored_map.astype(np.float32) * alpha) + (roi_background.astype(np.float32) * (1.0 - alpha))).astype(np.uint8)
        contours, _ = cv2.findContours(np.where(mask > 0, 255, 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_overlay, contours, -1, (248, 250, 236), 1)

        output = ((0.95 * full_image.astype(np.float32)) + 4.0).clip(0, 255).astype(np.uint8)
        if analysis_bbox is None:
            output = roi_overlay
        else:
            x1, y1, x2, y2 = analysis_bbox
            output[y1:y2, x1:x2] = roi_overlay
            cv2.rectangle(output, (x1, y1), (x2, y2), (244, 248, 232), 2)

        legend_h = 12
        legend_w = max(120, min(220, full_image.shape[1] // 3))
        legend_x = 14
        legend_y = full_image.shape[0] - 28
        legend = np.tile(np.linspace(0, 255, legend_w, dtype=np.uint8), (legend_h, 1))
        legend = cv2.applyColorMap(legend, cv2.COLORMAP_TURBO)
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        output[legend_y : legend_y + legend_h, legend_x : legend_x + legend_w] = legend
        cv2.rectangle(output, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (255, 255, 255), 1)
        cv2.putText(output, "0", (legend_x, legend_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(output, f"{lai_cap:.1f}", (legend_x + legend_w - 34, legend_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(output, f"LAI map | mean {lai_value:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
        cv2.putText(output, f"Conf: {confidence:.2f}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        result_img = Image.fromarray(output)
        buffer = io.BytesIO()
        result_img.save(buffer, format="JPEG", quality=86, optimize=True)
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

    def _build_recommendation(self, confidence_data: dict, status_info: dict, lai_result: dict, analysis_bbox):
        recommendation = confidence_data["recommendation"]
        recommendation += (
            f"\n\nМетод: {lai_result.get('method_version', CVEngine.SCIENTIFIC_METHOD)}"
            f"\nДоля просветов: {lai_result.get('gap_fraction', 0.0) * 100:.1f}%"
            f"\nЭффективный k: {lai_result.get('extinction_coefficient', 0.0):.2f}"
            f"\nОсвещение: {lai_result.get('illumination_class', 'неизвестно')}"
            f"\nLAI P10-P90: {lai_result.get('local_lai_p10', 0.0):.2f}-{lai_result.get('local_lai_p90', 0.0):.2f}"
        )

        if analysis_bbox is not None:
            recommendation += "\nОбнаружена выделенная рабочая зона кадра. Расчет выполнен внутри нее."

        status = status_info.get("status", "unknown")
        status_confidence = float(status_info.get("confidence", 0.0))
        if status != "unknown" and status_confidence >= 0.55:
            recommendation += f"\n\nAI-оценка: {status.title()} ({status_confidence * 100:.0f}%)"

            status_lower = status.lower()
            if any(keyword in status_lower for keyword in ("stressed", "yellowing", "drought", "deficiency")):
                recommendation += "\nОбнаружены возможные признаки стресса. Проверьте водообеспечение, влажность почвы и питание растений."
            elif any(keyword in status_lower for keyword in ("weed", "infested")):
                recommendation += "\nВероятно присутствует смешанная растительность. Проверьте участок на засоренность."
            elif any(keyword in status_lower for keyword in ("bare", "sparse", "early")):
                recommendation += "\nПлотность растительности низкая. Если нужен LAI именно для покрова, переснимите более плотный участок."

        return recommendation

    def _cleanup_memory(self, variables: list):
        for var in variables:
            if var is not None:
                del var
        gc.collect()
