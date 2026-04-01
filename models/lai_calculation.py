# -*- coding: utf-8 -*-
import io
import base64
import gc
import logging
import numpy as np
import cv2
from PIL import Image, ImageDraw

from odoo import models, fields, _
from .cv_engine import CVEngine

_logger = logging.getLogger(__name__)


class LAICalculation(models.Model):
    _name = 'lai.calculation'
    _description = 'LAI Calculation Record'

    name = fields.Char(string='Name', required=True)
    image = fields.Binary(string='Original Image', attachment=True)
    image_filename = fields.Char(string='Filename')
    
    crop_type = fields.Selection([
        ('wheat', 'Озимая пшеница'),
        ('corn', 'Кукуруза'),
        ('sunflower', 'Подсолнечник'),
        ('soy', 'Соя'),
        ('mixed', 'Смешанная'),
    ], string='Культура', default='wheat')

    lai_avg = fields.Float(string='Средний LAI', digits=(4, 2))
    lai_heatmap = fields.Binary(string='Heatmap', attachment=True)
    lai_heatmap_filename = fields.Char(string='Heatmap Filename')

    confidence = fields.Float(string='Уверенность модели', digits=(4, 3), default=0.0)
    coverage_percent = fields.Float(string='Покрытие растительностью %', digits=(4, 2))
    texture_homogeneity = fields.Float(string='Текстура (однородность)', digits=(4, 3))
    recommendation_text = fields.Text(string='Рекомендация системы')
    segmentation_method = fields.Char(string='Метод сегментации', default='CV_Ensemble_v2')

    user_id = fields.Many2one('res.users', string='User', default=lambda self: self.env.user)
    date_calculated = fields.Datetime(string='Calculated On', default=fields.Datetime.now)

    use_custom_calibration = fields.Boolean(string="Custom Calibration", default=False)
    custom_green_hue_center = fields.Float(string="Green Hue Center", default=0.17)
    custom_green_hue_width = fields.Float(string="Hue Sensitivity", default=3.0)
    custom_lai_min = fields.Float(string="Min LAI", default=0.5)
    custom_lai_max = fields.Float(string="Max LAI", default=6.0)

    def check_access_rights(self, operation, raise_exception=True):
        if self.env.user._is_public() and operation == 'read':
            return True
        return super().check_access_rights(operation, raise_exception)

    def _process_image_and_calculate_lai(self, image_data: bytes, crop_type: str):

        try:

            if not image_data:
                raise ValueError("Image data is empty")
            
            if len(image_data) < 100:
                raise ValueError(f"Image data too small: {len(image_data)} bytes")
            
            img_arr = None
            load_error = None
            
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as tmp:
                    tmp.write(image_data)
                    tmp_path = tmp.name
                
                img_bgr = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
                
                if img_bgr is not None:
                    img_arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    _logger.info(f"Image loaded via OpenCV: shape={img_arr.shape}")
                else:
                    load_error = "OpenCV failed to read image"
                
                os.unlink(tmp_path)
                
            except Exception as e:
                load_error = f"OpenCV error: {e}"
            
            if img_arr is None:
                try:
                    with io.BytesIO(image_data) as buf:
                        with Image.open(buf) as img_pil:
                            _logger.info(f"Image loaded via PIL: {img_pil.format}")
                            img_pil = img_pil.convert("RGB")
                            img_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                            img_arr = np.array(img_pil)
                except Exception as pil_err:
                    _logger.error(f"PIL failed: {pil_err}")
                    raise ValueError(f"Cannot process image: {load_error or pil_err}")
            
            if img_arr is None or img_arr.size == 0 or len(img_arr.shape) != 3:
                raise ValueError(f"Invalid image array: {img_arr.shape if img_arr is not None else 'None'}")
            
            cv_engine = CVEngine()
            image_proc = cv_engine.preprocess_image(img_arr)
            indices = cv_engine.calculate_vegetation_indices(image_proc)
            mask = cv_engine.ensemble_segmentation(indices)
            mask_refined = cv_engine.refine_mask_adaptive(mask, image_proc)
            
            lai_result = cv_engine.calculate_lai_from_mask(mask_refined, image_proc, crop_type=crop_type)
            confidence_data = cv_engine.estimate_confidence_v2(image_proc, mask_refined, lai_result)
            
            avg_lai = lai_result['lai_value']
            coverage = lai_result['coverage'] * 100
            texture_hom = lai_result['texture_homogeneity']
            confidence = confidence_data['confidence']
            recommendation = confidence_data['recommendation']
            
            heatmap_bytes = self._generate_heatmap_pil(image_proc, mask_refined, avg_lai)
            heatmap_filename = "lai_heatmap_cv2.png"
            
            del img_arr, image_proc, mask, mask_refined, indices
            gc.collect()
            
            return (avg_lai, heatmap_bytes, heatmap_filename, confidence, coverage, texture_hom, recommendation)
            
        except ValueError as ve:
            _logger.error(f"Validation error: {ve}")
            raise
        except Exception as e:
            _logger.exception("Unexpected error")
            raise ValueError(f"LAI calculation failed: {str(e)}")

    def _generate_heatmap_pil(self, image: np.ndarray, mask: np.ndarray, lai_value: float):
        """
        Генерация визуализации: изображение + полупрозрачная маска + текст
        """
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]
        
        heatmap = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        
        cv2.putText(
            heatmap, f'LAI: {lai_value:.2f}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            heatmap, f'Conf: {self.confidence:.2f}', (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        result_img = Image.fromarray(heatmap)
        
        buf = io.BytesIO()
        result_img.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        data = buf.getvalue()
        buf.close()
        result_img.close()
        
        return data