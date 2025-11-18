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
        return request.render('lai_estimator.lai_calculator_page')

    @http.route('/lai-calculate', type='http', auth='public', methods=['POST'], website=True, csrf=False)
    def lai_calculate(self, **kw):
        image_file = kw.get('image')
        if not image_file:
            return request.redirect('/lai-calculator?error=NoImage')

        if hasattr(image_file, 'content_length'):
            if image_file.content_length > MAX_IMAGE_SIZE:
                return request.redirect('/lai-calculator?error=ImageTooLarge')

        try:
            image_data = image_file.read()
            if not image_data:
                return request.redirect('/lai-calculator?error=EmptyImage')
            if len(image_data) > MAX_IMAGE_SIZE:
                return request.redirect('/lai-calculator?error=ImageTooLarge')

            crop_type = kw.get('crop_type', 'mixed')

            avg_lai, heatmap_bytes, heatmap_filename = request.env['lai.calculation'] \
                .sudo()._process_image_and_calculate_lai(image_data, crop_type)

            calc = request.env['lai.calculation'].sudo().create({
                'name': f'LAI-{request.env.user.name or "Guest"}',
                'image': base64.b64encode(image_data),
                'image_filename': image_file.filename,
                'crop_type': crop_type,
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
        return request.render('lai_estimator.lai_result_page', {'calc': calc})