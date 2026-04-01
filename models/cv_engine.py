import numpy as np
import cv2
from skimage.filters import threshold_otsu


class CVEngine:
    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения
        
        Args:
            image: RGB изображение в формате numpy array
            
        Returns:
            Обработанное изображение
        """

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        image_denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        lab = cv2.cvtColor(image_denoised, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image_corrected
    
    @staticmethod
    def calculate_vegetation_indices(image: np.ndarray) -> dict:
        """Расчёт ансамбля вегетационных индексов"""
        img_norm = image.astype(np.float32) / 255.0
        r = img_norm[:, :, 2]  # Red
        g = img_norm[:, :, 1]  # Green
        b = img_norm[:, :, 0]  # Blue
        
        epsilon = 1e-6
        
        # exG (excess Green)
        exg = 2 * g - r - b
        exg = np.clip(exg, -1, 1)
        
        # CIVE (Color Index of Vegetation Extraction)
        cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.787 / 255.0
        cive = np.clip(cive, -1, 1)  # Нормализация
        
        # VARI (Visible Atmospherically Resistant Index)
        vari_denom = (g + r - b + epsilon)
        vari = (g - r) / vari_denom
        vari = np.clip(vari, -1, 1)
        
        # GLI (Green Leaf Index)
        gli_denom = (2 * g + r + b + epsilon)
        gli = (2 * g - r - b) / gli_denom
        gli = np.clip(gli, -1, 1)
        
        # NDI (Normalized Difference Index)
        ndi_denom = (g + r + epsilon)
        ndi = (g - r) / ndi_denom
        ndi = np.clip(ndi, -1, 1)
        
        return {
            'exg': exg,
            'cive': cive,
            'vari': vari,
            'gli': gli,
            'ndi': ndi,
        }
    
    @staticmethod
    def ensemble_segmentation(indices: dict, weights: dict = None) -> np.ndarray:
        """
        Сегментация через взвешенное голосование индексов
        """
        if weights is None:
            weights = {
                'exg': 0.40,
                'cive': 0.25,
                'vari': 0.15,
                'gli': 0.15,
                'ndi': 0.05,
            }
        
        sample_index = indices['exg']
        vote_sum = np.zeros_like(sample_index, dtype=np.float32)
        total_weight = 0.0
        
        # exG
        exg_mask = indices['exg'] > 0.05
        vote_sum += weights['exg'] * exg_mask.astype(np.float32)
        total_weight += weights['exg']
        
        # CIVE
        cive_mask = indices['cive'] > 0.0
        vote_sum += weights['cive'] * cive_mask.astype(np.float32)
        total_weight += weights['cive']
        
        # VARI
        vari_normalized = np.clip(indices['vari'], -1, 1)
        vari_mask = vari_normalized > 0.1
        vote_sum += weights['vari'] * vari_mask.astype(np.float32)
        total_weight += weights['vari']
        
        # GLI
        gli_mask = indices['gli'] > 0.05
        vote_sum += weights['gli'] * gli_mask.astype(np.float32)
        total_weight += weights['gli']
        
        # NDI
        ndi_mask = indices['ndi'] > 0.1
        vote_sum += weights['ndi'] * ndi_mask.astype(np.float32)
        total_weight += weights['ndi']
        
        mask = (vote_sum > (total_weight * 0.5)).astype(np.uint8) * 255
        
        return mask
    
    @staticmethod
    def refine_mask_adaptive(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Улучшение маски с контролем расширения"""
        mask_refined = mask.copy()
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel_open)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel_close)
        
        min_area = int(image.shape[0] * image.shape[1] * 0.001)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_refined, connectivity=8
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask_refined[labels == i] = 0
        
        coverage = np.sum(mask_refined > 0) / mask_refined.size
        if coverage > 0.90:
            mask_refined = cv2.erode(mask_refined, kernel_open, iterations=1)
        
        return mask_refined

    @staticmethod
    def calculate_lai_from_mask(mask: np.ndarray, image: np.ndarray, crop_type: str = 'wheat') -> dict:
        """Расчет LAI с улучшенной текстурной обработкой"""
        from skimage.feature import graycomatrix, graycoprops
        
        coverage = np.sum(mask > 0) / mask.size
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if np.any(mask > 0):
            weighted_gray = gray.astype(np.float32) * (mask.astype(np.float32) / 255.0)
            weighted_gray = np.clip(weighted_gray, 0, 255).astype(np.uint8)
            
            gray_norm = cv2.normalize(weighted_gray, None, 0, 63, cv2.NORM_MINMAX).astype(np.uint8)
            
            try:
                glcm = graycomatrix(
                    gray_norm,
                    distances=[1, 2],
                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=64,
                    symmetric=True,
                    normed=True
                )
                
                contrast = float(np.mean(graycoprops(glcm, 'contrast')))
                homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
                energy = float(np.mean(graycoprops(glcm, 'energy')))
                
                homogeneity = np.clip(homogeneity * 10, 0, 1)
                energy = np.clip(energy * 5, 0, 1)
                
            except Exception as e:
                contrast, homogeneity, energy = 0.5, 0.5, 0.5
        else:
            contrast, homogeneity, energy = 0.0, 0.0, 0.0
        
        coefficients = {
            'wheat': {'a': 2.2, 'b': 0.6, 'c': 0.2},
            'soy': {'a': 2.5, 'b': 0.5, 'c': 0.3},
            'sunflower': {'a': 2.7, 'b': 0.4, 'c': 0.4},
            'corn': {'a': 2.8, 'b': 0.5, 'c': 0.3},
        }
        
        params = coefficients.get(crop_type, coefficients['wheat'])
        texture_factor = (homogeneity + energy) / 2
        
        if coverage > 0.8:
            lai_value = params['a'] * np.log(1 + coverage * 8) + params['b'] * texture_factor
        else:
            lai_value = params['a'] * coverage * 5 + params['b'] * texture_factor + params['c']
        
        lai_value = float(np.clip(lai_value, 0, 8))
        
        return {
            'lai_value': lai_value,
            'coverage': float(coverage),
            'texture_contrast': contrast,
            'texture_homogeneity': homogeneity,
            'texture_energy': energy,
            'texture_factor': texture_factor,
        }
    
    @staticmethod
    def estimate_confidence_v2(image: np.ndarray, mask: np.ndarray, 
                            lai_result: dict) -> dict:
        """
        Оценка уверенности с учетом LAI
        
        Args:
            image: Изображение
            mask: Маска растительности
            lai_result: Результаты расчета LAI
        
        Returns:
            Словарь с метриками уверенности
        """
        metrics = {}
        
        contrast_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
        metrics['contrast_score'] = np.clip(contrast_var / 1000, 0, 1)
        
        coverage = lai_result.get('coverage', 0)
        metrics['coverage_score'] = np.clip(coverage / 0.6, 0, 1)
        
        if np.any(mask > 0):
            mask_std = np.std(mask[mask > 0])
            metrics['mask_uniformity'] = float(1 - (mask_std / 255))
        else:
            metrics['mask_uniformity'] = 0.0
        
        texture_factor = lai_result.get('texture_factor', 0)
        metrics['texture_score'] = texture_factor
        
        edges = cv2.Canny(image, 100, 200)
        edge_ratio = float(np.sum(edges > 0) / edges.size)
        metrics['edge_penalty'] = np.clip(1 - (edge_ratio / 0.3), 0, 1)
        
        confidence = (
            0.25 * metrics['contrast_score'] +
            0.25 * metrics['coverage_score'] +
            0.20 * metrics['mask_uniformity'] +
            0.15 * metrics['texture_score'] +
            0.15 * metrics['edge_penalty']
        )
        
        flags = {
            'low_contrast': metrics['contrast_score'] < 0.3,
            'sparse_coverage': coverage < 0.1,
            'excessive_edges': edge_ratio > 0.25,
            'poor_texture': texture_factor < 0.2,
        }
        
        if confidence > 0.7:
            recommendation = "✅ Результат надёжен (точность ±0.5 LAI)"
        elif flags['low_contrast']:
            recommendation = "⚠️ Сделайте снимок при лучшем освещении (без теней)"
        elif flags['sparse_coverage']:
            recommendation = "⚠️ Мало растительности в кадре — увеличьте область"
        elif flags['excessive_edges']:
            recommendation = "⚠️ Возможно размытие — используйте штатив"
        elif flags['poor_texture']:
            recommendation = "⚠️ Низкая текстурная информативность"
        else:
            recommendation = "⚠️ Проверьте качество изображения"
        
        return {
            'confidence': float(confidence),
            'metrics': metrics,
            'flags': flags,
            'recommendation': recommendation,
            'estimated_error': float(0.3 + 0.7 * (1 - confidence)),
        }