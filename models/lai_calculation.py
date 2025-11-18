import io
import base64
import gc
import logging

import numpy as np
import cv2
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

from odoo import models, fields, _, api

_logger = logging.getLogger(__name__)


class LAICalculation(models.Model):
    _name = 'lai.calculation'
    _description = 'LAI Calculation Record'

    name = fields.Char(string='Name', required=True)
    image = fields.Binary(string='Original Image', attachment=True)
    image_filename = fields.Char(string='Filename')
    crop_type = fields.Selection([
        ('wheat', 'Wheat'),
        ('corn', 'Corn'),
        ('sunflower', 'Sunflower'),
        ('mixed', 'Mixed'),
    ], string='Crop Type', default='mixed')
    lai_avg = fields.Float(string='Average LAI', digits=(4, 2))
    lai_heatmap = fields.Binary(string='Heatmap Image', attachment=True)
    lai_heatmap_filename = fields.Char(string='Heatmap Filename')
    user_id = fields.Many2one('res.users', string='User', default=lambda self: self.env.user)
    date_calculated = fields.Datetime(string='Calculated On', default=fields.Datetime.now)

    def check_access_rights(self, operation, raise_exception=True):
        if self.env.user._is_public() and operation == 'read':
            return True
        return super().check_access_rights(operation, raise_exception)

    @api.model
    def _process_image_and_calculate_lai(self, image_data: bytes, crop_type: str):
        try:
            with io.BytesIO(image_data) as buf:
                with Image.open(buf).convert("RGB") as img_pil:
                    img_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    img_arr = np.array(img_pil)

            lai_map = self._generate_lai_map_from_color_segments(img_arr)

            del img_arr
            gc.collect()

            avg_lai = float(np.nanmean(lai_map))

            heatmap_bytes = self._generate_heatmap_pil(lai_map)
            heatmap_filename = "lai_heatmap.png"

            del lai_map
            gc.collect()

            return avg_lai, heatmap_bytes, heatmap_filename

        except Exception as e:
            _logger.exception("Error in _process_image_and_calculate_lai")
            raise

    def _generate_lai_map_from_color_segments(self, rgb_image, n_clusters=4):
        h, w, _ = rgb_image.shape
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv.reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = kmeans.fit_predict(hsv_pixels)

        lai_map = np.full((h, w), np.nan, dtype=np.float32)
        for i in range(n_clusters):
            mask = (labels.reshape(h, w) == i)
            if not np.any(mask):
                continue
            cluster_hsv = hsv[mask]
            avg_hue = np.mean(cluster_hsv[:, 0]) / 180.0
            avg_sat = np.mean(cluster_hsv[:, 1]) / 255.0
            avg_val = np.mean(cluster_hsv[:, 2]) / 255.0
            green_score = max(0.0, 1.0 - abs(avg_hue - 0.17) * 3.0)
            lai_val = 0.5 + 4.5 * green_score * avg_sat * avg_val
            lai_map[mask] = np.clip(lai_val, 0.0, 6.0)

        return lai_map

    def _generate_heatmap_pil(self, lai_map):
        h, w = lai_map.shape
        palette = [
            (102, 51, 26),
            (153, 102, 51),
            (204, 179, 77),
            (102, 204, 102),
            (51, 153, 51),
            (0, 102, 0),
        ]

        normalized = np.clip(lai_map, 0, 6)
        indices = (normalized * (len(palette) - 1) / 6.0).astype(np.uint8)

        heatmap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i, color in enumerate(palette):
            heatmap_rgb[indices == i] = color

        result_img = Image.fromarray(heatmap_rgb)
        draw = ImageDraw.Draw(result_img)
        draw.text((10, 10), "LAI: 0–6", fill=(255, 255, 255))

        buf = io.BytesIO()
        result_img.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        data = buf.getvalue()
        buf.close()
        result_img.close()
        del heatmap_rgb, indices, normalized
        return data