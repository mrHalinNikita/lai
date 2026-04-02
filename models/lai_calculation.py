import io
import base64
import gc
import logging
import numpy as np
import cv2
from PIL import Image
import time
import os

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
    
    segmentation_method = fields.Selection([
        ('cv_ensemble', 'CV Ensemble (OpenCV + Indices)'),
        ('sam', 'Segment Anything Model (SAM vit_b)'),
    ], string='Метод сегментации', default='cv_ensemble')

    crop_status = fields.Char(string="AI Статус посевов", readonly=True)
    status_confidence = fields.Float(string="Уверенность AI", digits=(4, 3), readonly=True)

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

        img_arr = image_proc = mask = mask_refined = indices = None
        lai_result = confidence_data = None
        cv_engine = CVEngine()
        
        try:
            img_arr = self._load_image(image_data)
            if img_arr is None or img_arr.size == 0:
                raise ValueError("Failed to load or process image")
            
            method = self.segmentation_method or 'cv_ensemble'
            
            if method == 'sam':
                image_proc, mask_refined = self._segment_with_sam_pipeline(cv_engine, img_arr)
            else:
                image_proc, mask_refined = self._segment_with_cv_pipeline(cv_engine, img_arr)
            
            lai_result = cv_engine.calculate_lai_from_mask(mask_refined, image_proc, crop_type=crop_type)
            confidence_data = cv_engine.estimate_confidence_v2(image_proc, mask_refined, lai_result)
            
            avg_lai = float(lai_result['lai_value'])
            coverage = float(lai_result['coverage'] * 100)
            texture_hom = float(lai_result['texture_homogeneity'])
            confidence = float(confidence_data['confidence'])
            recommendation = confidence_data['recommendation']
            heatmap_bytes = self._generate_heatmap_pil(image_proc, mask_refined, avg_lai)

            status_info = self._assess_crop_status_with_clip(img_arr)
            if status_info['status'] != 'unknown' and status_info['confidence'] > 0.3:
                recommendation += f"\n\n🤖 AI Assessment: {status_info['status'].title()} ({status_info['confidence']*100:.0f}%)"
                
                status_lower = status_info['status'].lower()
                if any(kw in status_lower for kw in ['stressed', 'yellowing', 'drought', 'deficiency']):
                    recommendation += "\nОбнаружены признаки стресса. Проверьте полив, влажность почвы и уровень питательных веществ (N-P-K)."
                elif any(kw in status_lower for kw in ['weed', 'infested']):
                    recommendation += "\nВысокая вероятность появления сорняков. Рассмотрите возможность целенаправленного применения гербицидов или механической прополки."
                elif any(kw in status_lower for kw in ['bare', 'sparse', 'early']):
                    recommendation += "\nНизкая плотность растительности. Проверьте всхожесть или рассмотрите возможность повторного посева, если она ниже порогового значения."
            
            _logger.info(f"LAI calculated: method={method}, lai={avg_lai:.2f}, coverage={coverage:.1f}%, conf={confidence:.3f}")
            
            return (avg_lai, heatmap_bytes, "lai_heatmap.png", confidence, coverage, texture_hom, recommendation)
            
        except ValueError:
            raise
        except Exception as e:
            _logger.exception("Unexpected error in LAI calculation")
            raise ValueError(f"LAI calculation failed: {str(e)}")
        finally:
            self._cleanup_memory([img_arr, image_proc, mask, mask_refined, indices, lai_result, confidence_data])
    
    def _load_image(self, image_data: bytes) -> np.ndarray | None:
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name
            
            img_bgr = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
            os.unlink(tmp_path)
            
            if img_bgr is not None:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            _logger.debug(f"OpenCV load failed: {e}")
        
        try:
            with io.BytesIO(image_data) as buf:
                with Image.open(buf) as img_pil:
                    img_pil = img_pil.convert("RGB")
                    img_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    return np.array(img_pil)
        except Exception as e:
            _logger.error(f"PIL load failed: {e}")
            return None
    
    def _segment_with_sam_pipeline(self, cv_engine, img_arr: np.ndarray):
        from .sam_adapter import SAMAdapter
        
        start_time = time.time()
        _logger.info(f"[SAM vit_b] Starting segmentation")
        
        try:
            checkpoint_path = os.environ.get('SAM_CHECKPOINT_PATH', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sam_checkpoints', 'sam_vit_b_01ec64.pth'))
            device = 'cuda' if os.environ.get('USE_CUDA') == '1' else 'cpu'
            
            SAMAdapter.initialize(checkpoint_path, device=device)
            
            h, w = img_arr.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                img_resized = cv2.resize(img_arr, new_size, interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = img_arr.copy()
            
            mask = SAMAdapter.segment_vegetation(
                img_resized,
                prompt_box=None,
                min_area_ratio=0.002
            )
            
            if img_resized.shape != img_arr.shape[:2]:
                mask = cv2.resize(mask, (img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            image_proc = cv_engine.preprocess_image(img_arr)
            
            elapsed = time.time() - start_time
            coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100
            _logger.info(f"[SAM vit_b] Done in {elapsed:.2f}s | coverage: {coverage:.1f}%")
            
            return image_proc, mask
            
        except Exception as e:
            _logger.warning(f"[SAM] Failed: {e}. Falling back to CV_Ensemble...")
            return self._segment_with_cv_pipeline(cv_engine, img_arr)
        finally:
            SAMAdapter.aggressive_cleanup()
            if 'img_resized' in locals() and img_resized is not img_arr:
                del img_resized
            gc.collect()
    
    def _segment_with_cv_pipeline(self, cv_engine, img_arr: np.ndarray):
        start_time = time.time()
        
        image_proc = cv_engine.preprocess_image(img_arr)
        indices = cv_engine.calculate_vegetation_indices(image_proc)
        mask = cv_engine.ensemble_segmentation(indices)
        mask_refined = cv_engine.refine_mask_adaptive(mask, image_proc)
        
        elapsed = time.time() - start_time
        coverage = np.sum(mask_refined > 0) / (mask_refined.size) * 100
        _logger.info(f"[CV_Ensemble] Done in {elapsed:.2f}s | coverage: {coverage:.1f}%")
        
        return image_proc, mask_refined
    
    def _generate_heatmap_pil(self, image: np.ndarray, mask: np.ndarray, lai_value: float) -> bytes:
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]
        heatmap = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        
        cv2.putText(heatmap, f'LAI: {lai_value:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(heatmap, f'Conf: {self.confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        result_img = Image.fromarray(heatmap)
        buf = io.BytesIO()
        result_img.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        
        data = buf.getvalue()
        buf.close()
        result_img.close()
        return data

    def _assess_crop_status_with_clip(self, rgb_image: np.ndarray) -> dict:

        from .clip_adapter import CLIPAdapter
        
        try:
            result = CLIPAdapter.assess_crop_status(rgb_image)
            
            self.write({
                'crop_status': result['status'],
                'status_confidence': result['confidence']
            })
            
            return result
        except Exception as e:
            _logger.warning(f"CLIP assessment skipped: {e}")
            return {'status': 'unknown', 'confidence': 0.0, 'all_probabilities': {}}
        finally:
            CLIPAdapter.cleanup()
    
    def _cleanup_memory(self, variables: list):
        for var in variables:
            if var is not None:
                del var
        gc.collect()