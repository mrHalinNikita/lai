from odoo import http
from odoo.http import request
import base64
import logging

_logger = logging.getLogger(__name__)


class LAICalculatorController(http.Controller):

    @http.route('/lai-calculator', type='http', auth='public', website=True)
    def lai_calculator(self, **kw):
        return request.render('lai_estimator.lai_calculator_page', {})

    @http.route('/lai-calculate', type='http', auth='public', methods=['POST'], website=True, csrf=False)
    def lai_calculate(self, **kw):
        if 'image' not in kw or not kw['image']:
            return request.redirect('/lai-calculator?error=NoImage')

        image_file = kw['image']
        crop_type = kw.get('crop_type', 'mixed')

        image_data = image_file.read()

        if len(image_data) == 0:
            return request.redirect('/lai-calculator?error=EmptyImage')

        if len(image_data) > 10 * 1024 * 1024:
            return request.redirect('/lai-calculator?error=ImageTooLarge')

        try:
            calc = request.env['lai.calculation'].sudo().create({
                'name': 'LAI Calculation',
                'image': base64.b64encode(image_data),
                'image_filename': image_file.filename,
                'crop_type': crop_type,
            })

            avg_lai, heatmap_bytes, heatmap_filename = calc.calculate_lai_from_image(image_data, crop_type)

            if not isinstance(heatmap_bytes, bytes) or len(heatmap_bytes) == 0:
                raise ValueError("Heatmap generation returned empty or invalid data.")

            calc.sudo().write({
                'lai_avg': avg_lai,
                'lai_heatmap': base64.b64encode(heatmap_bytes),
                'lai_heatmap_filename': heatmap_filename,
            })

            return request.redirect(f'/lai-result/{calc.id}')

        except Exception as e:
            _logger.exception("LAI calculation failed")
            return request.redirect(f'/lai-calculator?error={str(e)}')

    @http.route('/lai-result/<int:calc_id>', type='http', auth='public', website=True)
    def lai_result(self, calc_id, **kw):
        calc = request.env['lai.calculation'].sudo().browse(calc_id)
        if not calc.exists():
            return request.redirect('/lai-calculator?error=NotFound')
        return request.render('lai_estimator.lai_result_page', {'calc': calc})