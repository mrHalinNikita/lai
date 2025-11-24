import logging
from werkzeug.exceptions import RequestEntityTooLarge
from odoo import http
from odoo.http import request
import base64

_logger = logging.getLogger(__name__)
MAX_IMAGE_SIZE = 5 * 1024 * 1024


class LAICalculatorController(http.Controller):

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
            # Передаём параметры обратно при ошибке
            return request.redirect(
                '/lai-calculator?error=ProcessingFailed'
                + ''.join(f'&{k}={v}' for k, v in kw.items() if k in [
                    'use_custom_calibration',
                    'custom_green_hue_center',
                    'custom_green_hue_width',
                    'custom_lai_min',
                    'custom_lai_max',
                    'crop_type'
                ])
            )

    @http.route('/lai-result/<int:calc_id>', type='http', auth='public', website=True)
    def lai_result(self, calc_id, **kw):
        calc = request.env['lai.calculation'].sudo().browse(calc_id)
        if not calc.exists():
            return request.redirect('/lai-calculator?error=NotFound')
        return request.render('lai_estimator.lai_result_page', {'calc': calc})