from odoo import models, fields
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import io


class LAICalculation(models.Model):
    _name = 'lai.calculation'
    _description = 'LAI Calculation Record'

    name = fields.Char(string='Name', required=True, default=lambda self: f'LAI-{self.id or 0}')
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

    def calculate_lai_from_image(self, image_data, crop_type):
        img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_arr = np.array(img_pil)
        lai_map = self._generate_lai_map_from_color_segments(img_arr)
        avg_lai = float(np.nanmean(lai_map))
        heatmap_bytes = self._generate_heatmap_overlay(img_arr, lai_map)
        return avg_lai, heatmap_bytes, f"lai_heatmap_{self.id or 0}.png"

    def _generate_lai_map_from_color_segments(self, rgb_image, n_clusters=5):
        h, w, _ = rgb_image.shape
        hsv = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv.reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(hsv_pixels)

        lai_map = np.zeros((h, w), dtype=np.float32)

        for i in range(n_clusters):
            mask = (labels.reshape(h, w) == i)
            if not np.any(mask):
                continue

            cluster_hsv = hsv[mask]
            avg_hue = np.mean(cluster_hsv[:, 0])
            avg_sat = np.mean(cluster_hsv[:, 1])
            avg_val = np.mean(cluster_hsv[:, 2])

            hue_norm = avg_hue / 180.0
            green_score = max(0, 1 - abs(hue_norm - 0.17) * 3)
            sat_score = avg_sat / 255.0
            val_score = avg_val / 255.0

            lai_cluster = 0.5 + 5.0 * green_score * sat_score * val_score
            lai_map[mask] = min(6.0, max(0.0, lai_cluster))

        return lai_map

    def _generate_heatmap_overlay(self, rgb_image, lai_map, alpha=0.75):
        colors = [
            (0.4, 0.2, 0.1),
            (0.6, 0.4, 0.2),
            (0.8, 0.7, 0.3),
            (0.4, 0.8, 0.4),
            (0.2, 0.6, 0.2),
            (0.0, 0.4, 0.0),
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("lai_colormap", colors, N=100)
        lai_viz = np.clip(lai_map, 0, 6)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(rgb_image)
        im = ax.imshow(lai_viz, cmap=cmap, vmin=0, vmax=6, alpha=alpha)
        contours = ax.contour(lai_viz, levels=np.arange(0.5, 6.0, 0.5), colors='white', linewidths=0.4, alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=6, fmt="%.1f")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("LAI (m²/m²)", rotation=270, labelpad=20)
        cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])

        ax.set_title("LAI Spatial Map — Central Black Soil Region", fontsize=14, weight='bold')
        ax.axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        plt.close()

        return buf.read()