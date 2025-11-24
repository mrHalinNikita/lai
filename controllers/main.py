import logging
from werkzeug.exceptions import RequestEntityTooLarge
from odoo import http
from odoo.http import request
from odoo.tools import DEFAULT_SERVER_DATETIME_FORMAT
import base64

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
        })

    @http.route('/lai-calculate', type='http', auth='public', methods=['POST'], website=True, csrf=False)
    def lai_calculate(self, **kw):
        image_file = kw.get('image')
        if not image_file:
            return request.redirect('/lai-calculator?error=NoImage')

        if hasattr(image_file, 'content_length'):
            if image_file.content_length > MAX_IMAGE_SIZE:
                return request.redirect('/lai-calculator?error=ImageTooLarge')

        def safe_float(val, default):
            try:
                return float(val) if val not in (None, '') else default
            except (ValueError, TypeError):
                return default

        try:
            image_data = image_file.read()
            if not image_data:
                return request.redirect('/lai-calculator?error=EmptyImage')
            if len(image_data) > MAX_IMAGE_SIZE:
                return request.redirect('/lai-calculator?error=ImageTooLarge')

            crop_type = kw.get('crop_type', 'mixed')
            use_custom = bool(kw.get('use_custom_calibration'))

            if use_custom:
                is_valid, error_msg = self._validate_calibration_params(kw)
                if not is_valid:
                    params = {k: v for k, v in kw.items() if k != 'image' and v is not None}
                    query = '&'.join(f'{k}={v}' for k, v in params.items())
                    return request.redirect(f'/lai-calculator?error={error_msg}&{query}')

            calc_vals = {
                'name': f'LAI-{request.env.user.name or "Guest"}',
                'image': base64.b64encode(image_data),
                'image_filename': image_file.filename,
                'crop_type': crop_type,
                'use_custom_calibration': use_custom,
                'custom_green_hue_center': safe_float(kw.get('custom_green_hue_center'), 0.17),
                'custom_green_hue_width': safe_float(kw.get('custom_green_hue_width'), 3.0),
                'custom_lai_min': safe_float(kw.get('custom_lai_min'), 0.5),
                'custom_lai_max': safe_float(kw.get('custom_lai_max'), 6.0),
            }

            calc = request.env['lai.calculation'].sudo().create(calc_vals)
            avg_lai, heatmap_bytes, heatmap_filename = calc._process_image_and_calculate_lai(image_data, crop_type)

            calc.write({
                'lai_avg': avg_lai,
                'lai_heatmap': base64.b64encode(heatmap_bytes),
                'lai_heatmap_filename': heatmap_filename,
            })

            return request.redirect(f'/lai-result/{calc.id}')

        except MemoryError:
            _logger.error("MemoryError during LAI processing")
            return request.redirect('/lai-calculator?error=ServerOverloaded')
        except RequestEntityTooLarge:
            return request.redirect('/lai-calculator?error=ImageTooLarge')
        except Exception as e:
            _logger.exception("Unexpected error in LAI calculation")
            return request.redirect('/lai-calculator?error=ProcessingFailed')
    
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