import logging
from werkzeug.exceptions import RequestEntityTooLarge
from odoo import http
from odoo.http import request
from odoo.tools import DEFAULT_SERVER_DATETIME_FORMAT
import base64
import time

_logger = logging.getLogger(__name__)
MAX_IMAGE_SIZE = 5 * 1024 * 1024


class LAICalculatorController(http.Controller):

    def _validate_calibration_params(self, params):
        try:
            hue_center = float(params.get('custom_green_hue_center', 0.17))
            hue_width = float(params.get('custom_green_hue_width', 3.0))
            lai_min = float(params.get('custom_lai_min', 0.5))
            lai_max = float(params.get('custom_lai_max', 6.0))
        except (TypeError, ValueError):
            return False, "Invalid number format in calibration parameters."

        if not (0.0 <= hue_center <= 1.0):
            return False, "Green Hue Center must be between 0 and 1."
        if not (0.1 <= hue_width <= 10.0):
            return False, "Hue Sensitivity must be between 0.1 and 10."
        if not (0.0 <= lai_min <= 5.9):
            return False, "Min LAI must be between 0 and 5.9."
        if not (0.1 <= lai_max <= 10.0):
            return False, "Max LAI must be between 0.1 and 10."
        if lai_min >= lai_max:
            return False, "Min LAI must be less than Max LAI."

        return True, ""

    @http.route('/lai-calculator', type='http', auth='public', website=True)
    def lai_calculator(self, **kw):
        return request.render('lai_estimator.lai_calculator_page', {
            'use_custom_calibration': kw.get('use_custom_calibration') == 'on',
            'custom_green_hue_center': kw.get('custom_green_hue_center', '0.17'),
            'custom_green_hue_width': kw.get('custom_green_hue_width', '3.0'),
            'custom_lai_min': kw.get('custom_lai_min', '0.5'),
            'custom_lai_max': kw.get('custom_lai_max', '6.0'),
            'segmentation_method': kw.get('segmentation_method', 'cv_ensemble'),
        })

    @http.route('/lai-calculate', type='http', auth='public', methods=['POST'], website=True, csrf=False)
    def lai_calculate(self, **kw):
        image_file = kw.get('image')
        
        if not image_file:
            _logger.warning("No image file in request")
            return request.redirect('/lai-calculator?error=NoImage')
        
        try:
            image_data = image_file.read()
        except Exception as e:
            _logger.error(f"Failed to read image file: {e}")
            return request.redirect('/lai-calculator?error=ReadError')
        
        if not image_data or len(image_data) == 0:
            _logger.warning("Empty image data received")
            return request.redirect('/lai-calculator?error=EmptyImage')
        
        if len(image_data) > MAX_IMAGE_SIZE:
            _logger.warning(f"Image too large: {len(image_data)} bytes")
            return request.redirect('/lai-calculator?error=ImageTooLarge')
        
        _logger.info(f"Received image: {len(image_data)} bytes, filename: {image_file.filename}")
        _logger.info(f"First 20 bytes (hex): {image_data[:20].hex()}")
        
        if image_file.filename:
            ext = image_file.filename.lower().split('.')[-1]
            valid_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']
            if ext not in valid_extensions:
                _logger.warning(f"Invalid file extension: {ext}")
                return request.redirect(f'/lai-calculator?error=InvalidFormat&ext={ext}')
        
        def safe_float(val, default):
            try:
                return float(val) if val not in (None, '') else default
            except (ValueError, TypeError):
                return default
        
        crop_type = kw.get('crop_type', 'wheat')
        use_custom = bool(kw.get('use_custom_calibration'))
        
        segmentation_method = kw.get('segmentation_method', 'cv_ensemble')
        valid_methods = ['cv_ensemble', 'sam']
        if segmentation_method not in valid_methods:
            _logger.warning(f"Invalid segmentation_method: {segmentation_method}, falling back to cv_ensemble")
            segmentation_method = 'cv_ensemble'
        
        if use_custom:
            is_valid, error_msg = self._validate_calibration_params(kw)

            if not is_valid:
                params = {k: v for k, v in kw.items() if k != 'image' and v is not None}
                query = '&'.join(f'{k}={v}' for k, v in params.items())
                return request.redirect(f'/lai-calculator?error={error_msg}&{query}')
        
        try:
            calc_vals = {
                'name': f'LAI-{request.env.user.name or "Guest"}-{int(time.time())}',
                'image': base64.b64encode(image_data).decode('utf-8'),
                'image_filename': image_file.filename,
                'crop_type': crop_type,
                'segmentation_method': segmentation_method,
                'use_custom_calibration': use_custom,
                'custom_green_hue_center': safe_float(kw.get('custom_green_hue_center'), 0.17),
                'custom_green_hue_width': safe_float(kw.get('custom_green_hue_width'), 3.0),
                'custom_lai_min': safe_float(kw.get('custom_lai_min'), 0.5),
                'custom_lai_max': safe_float(kw.get('custom_lai_max'), 6.0),
            }
            
            calc = request.env['lai.calculation'].sudo().create(calc_vals)
            
            result = calc._process_image_and_calculate_lai(image_data, crop_type)
            
            avg_lai, heatmap_bytes, heatmap_filename, confidence, coverage, texture_hom, recommendation = result
            
            calc.write({
                'lai_avg': avg_lai,
                'lai_heatmap': base64.b64encode(heatmap_bytes).decode('utf-8'),
                'lai_heatmap_filename': heatmap_filename,
                'confidence': confidence,
                'coverage_percent': coverage,
                'texture_homogeneity': texture_hom,
                'recommendation_text': recommendation,
            })
            
            _logger.info(f"LAI calculation successful: LAI={avg_lai}, Confidence={confidence}")
            
            return request.redirect(f'/lai-result/{calc.id}')
            
        except ValueError as ve:
            _logger.error(f"Validation error: {ve}")
            safe_msg = str(ve).replace('\n', ' ').replace('\r', '')[:150]
            return request.redirect(f'/lai-calculator?error=ValidationError&msg={safe_msg}')
        except MemoryError:
            _logger.error("MemoryError during LAI processing")
            return request.redirect('/lai-calculator?error=ServerOverloaded')
        except Exception as e:
            _logger.exception("Unexpected error in LAI calculation")
            return request.redirect(f'/lai-calculator?error=ProcessingFailed&msg={str(e)}')
    
    @http.route('/lai-result/<int:calc_id>', type='http', auth='public', website=True)
    def lai_result(self, calc_id, **kw):
        calc = request.env['lai.calculation'].sudo().browse(calc_id)
        if not calc.exists():
            return request.redirect('/lai-calculator?error=NotFound')

        date_iso = (calc.date_calculated.isoformat() + 'Z') if calc.date_calculated else None
        
        return request.render('lai_estimator.lai_result_page', {
            'calc': calc,
            'date_iso': date_iso
        })