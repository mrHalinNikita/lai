import cv2
import numpy as np


class CVEngine:
    WORKING_MAX_DIM = 960
    TEXTURE_MAX_DIM = 192

    @staticmethod
    def downscale_for_processing(image: np.ndarray, max_dim: int | None = None) -> tuple[np.ndarray, float]:
        max_dim = max_dim or CVEngine.WORKING_MAX_DIM
        height, width = image.shape[:2]
        largest_dim = max(height, width)

        if largest_dim <= max_dim:
            return image, 1.0

        scale = max_dim / float(largest_dim)
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Median blur is much cheaper than bilateral filtering for this use case.
        image_denoised = cv2.medianBlur(image, 3)

        lab = cv2.cvtColor(image_denoised, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2RGB)

    @staticmethod
    def calculate_vegetation_indices(image: np.ndarray) -> dict:
        img_norm = image.astype(np.float32) / 255.0
        r = img_norm[:, :, 0]
        g = img_norm[:, :, 1]
        b = img_norm[:, :, 2]

        epsilon = 1e-6
        exg = np.clip((2.0 * g) - r - b, -1.0, 1.0)
        cive = np.clip((0.441 * r) - (0.811 * g) + (0.385 * b) + (18.787 / 255.0), -1.0, 1.0)
        vari = np.clip((g - r) / (g + r - b + epsilon), -1.0, 1.0)
        gli = np.clip(((2.0 * g) - r - b) / ((2.0 * g) + r + b + epsilon), -1.0, 1.0)
        ndi = np.clip((g - r) / (g + r + epsilon), -1.0, 1.0)

        return {
            "exg": exg,
            "cive": cive,
            "vari": vari,
            "gli": gli,
            "ndi": ndi,
        }

    @staticmethod
    def segment_green_by_hsv(
        image: np.ndarray,
        hue_center: float = 0.17,
        hue_width: float = 3.0,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue = int(np.clip(hue_center, 0.0, 1.0) * 179)
        width = max(2, int(np.clip(hue_width, 0.1, 10.0) * 4))

        lower = np.array([max(0, hue - width), 35, 25], dtype=np.uint8)
        upper = np.array([min(179, hue + width), 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    @staticmethod
    def estimate_background_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        height, width = image.shape[:2]
        border = max(6, int(min(height, width) * 0.08))

        border_mask = np.zeros((height, width), dtype=np.uint8)
        border_mask[:border, :] = 255
        border_mask[-border:, :] = 255
        border_mask[:, :border] = 255
        border_mask[:, -border:] = 255

        low_sat = hsv[:, :, 1] < 70
        candidate_mask = (border_mask > 0) & low_sat
        if np.count_nonzero(candidate_mask) < 64:
            candidate_mask = border_mask > 0

        background_pixels = lab[candidate_mask]
        background_color = np.median(background_pixels, axis=0).astype(np.float32)
        distance = np.linalg.norm(lab.astype(np.float32) - background_color, axis=2)

        return distance, candidate_mask.astype(np.uint8) * 255

    @staticmethod
    def segment_colored_foreground(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        distance, _ = CVEngine.estimate_background_mask(image)

        saturation_mask = hsv[:, :, 1] > 45
        value_mask = hsv[:, :, 2] < 245
        distance_mask = distance > max(18.0, float(np.percentile(distance, 65)))
        colored = saturation_mask & value_mask & distance_mask
        return np.where(colored, 255, 0).astype(np.uint8)

    @staticmethod
    def normalize_map(values: np.ndarray, lower_q: float = 5.0, upper_q: float = 95.0) -> np.ndarray:
        lower = float(np.percentile(values, lower_q))
        upper = float(np.percentile(values, upper_q))
        if upper <= lower:
            return np.zeros_like(values, dtype=np.float32)
        normalized = (values.astype(np.float32) - lower) / (upper - lower)
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def build_vegetation_score(indices: dict, image: np.ndarray, crop_type: str = "mixed") -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        distance, _ = CVEngine.estimate_background_mask(image)

        sat_score = hsv[:, :, 1].astype(np.float32) / 255.0
        val_score = 1.0 - (hsv[:, :, 2].astype(np.float32) / 255.0)
        bg_score = CVEngine.normalize_map(distance, 10.0, 90.0)
        exg_score = CVEngine.normalize_map(indices["exg"], 10.0, 95.0)
        vari_score = CVEngine.normalize_map(indices["vari"], 10.0, 95.0)
        gli_score = CVEngine.normalize_map(indices["gli"], 10.0, 95.0)

        hue = hsv[:, :, 0].astype(np.float32)
        hue_distance = np.minimum(np.abs(hue - 60.0), 180.0 - np.abs(hue - 60.0)) / 60.0
        hue_score = np.clip(1.0 - hue_distance, 0.0, 1.0)

        if crop_type == "vertical":
            score = (
                0.24 * bg_score
                + 0.20 * exg_score
                + 0.16 * vari_score
                + 0.10 * gli_score
                + 0.12 * sat_score
                + 0.08 * val_score
                + 0.10 * hue_score
            )
        else:
            score = (
                0.20 * bg_score
                + 0.24 * exg_score
                + 0.18 * vari_score
                + 0.12 * gli_score
                + 0.14 * sat_score
                + 0.12 * hue_score
            )

        return np.clip(score.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def ensemble_segmentation(
        indices: dict,
        image: np.ndarray,
        hue_center: float = 0.17,
        hue_width: float = 3.0,
        crop_type: str = "mixed",
        vegetation_score: np.ndarray | None = None,
        weights: dict | None = None,
    ) -> np.ndarray:
        vegetation_score = vegetation_score if vegetation_score is not None else CVEngine.build_vegetation_score(indices, image, crop_type=crop_type)
        hsv_mask = CVEngine.segment_green_by_hsv(image, hue_center=hue_center, hue_width=hue_width) > 0
        foreground_mask = CVEngine.segment_colored_foreground(image) > 0

        likely_pixels = vegetation_score[foreground_mask]
        if likely_pixels.size < 64:
            likely_pixels = vegetation_score.reshape(-1)

        score_uint8 = np.clip(likely_pixels * 255.0, 0, 255).astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(score_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        percentile_threshold = float(np.percentile(likely_pixels, 60 if crop_type == "vertical" else 55))
        adaptive_threshold = max(float(otsu_threshold) / 255.0, percentile_threshold)
        adaptive_threshold = float(np.clip(adaptive_threshold, 0.26, 0.68))

        auxiliary_mask = (
            (vegetation_score >= adaptive_threshold)
            | ((vegetation_score >= max(0.24, adaptive_threshold - 0.08)) & hsv_mask)
        )
        auxiliary_mask &= foreground_mask | hsv_mask

        mask = np.where(auxiliary_mask, 255, 0).astype(np.uint8)
        return CVEngine.refine_mask_adaptive(mask, image, crop_type=crop_type)

    @staticmethod
    def refine_mask_adaptive(mask: np.ndarray, image: np.ndarray, crop_type: str = "mixed") -> np.ndarray:
        refined = mask
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)

        min_area_ratio = 0.0009 if crop_type == "vertical" else 0.0005
        min_area = max(24, int(image.shape[0] * image.shape[1] * min_area_ratio))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
        cleaned = np.zeros_like(refined)

        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == idx] = 255

        coverage = float(np.count_nonzero(cleaned) / cleaned.size)
        if coverage > 0.92:
            cleaned = cv2.erode(cleaned, kernel_open, iterations=1)

        return cleaned

    @staticmethod
    def extract_mask_features(mask: np.ndarray) -> dict:
        if not np.any(mask):
            return {
                "component_count": 0,
                "largest_component_ratio": 0.0,
                "coverage": 0.0,
            }

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
        mask_area = float(np.count_nonzero(mask))
        largest_ratio = float(component_areas.max() / mask_area) if component_areas.size else 0.0

        return {
            "component_count": int(component_areas.size),
            "largest_component_ratio": largest_ratio,
            "coverage": float(mask_area / mask.size),
        }

    @staticmethod
    def extract_texture_features(image: np.ndarray, mask: np.ndarray) -> dict:
        from skimage.feature import graycomatrix, graycoprops

        if not np.any(mask):
            return {
                "texture_contrast": 0.0,
                "texture_homogeneity": 0.0,
                "texture_energy": 0.0,
                "texture_factor": 0.0,
            }

        y_indices, x_indices = np.where(mask > 0)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        gray = cv2.cvtColor(image[y_min : y_max + 1, x_min : x_max + 1], cv2.COLOR_RGB2GRAY)
        local_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

        gray = cv2.bitwise_and(gray, gray, mask=local_mask)
        gray, _ = CVEngine.downscale_for_processing(gray, max_dim=CVEngine.TEXTURE_MAX_DIM)
        gray = cv2.normalize(gray, None, 0, 31, cv2.NORM_MINMAX).astype(np.uint8)

        try:
            glcm = graycomatrix(
                gray,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2],
                levels=32,
                symmetric=True,
                normed=True,
            )
            contrast = float(np.mean(graycoprops(glcm, "contrast")))
            homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
            energy = float(np.mean(graycoprops(glcm, "energy")))
        except Exception:
            contrast, homogeneity, energy = 0.0, 0.0, 0.0

        return {
            "texture_contrast": contrast,
            "texture_homogeneity": float(np.clip(homogeneity, 0.0, 1.0)),
            "texture_energy": float(np.clip(energy, 0.0, 1.0)),
            "texture_factor": float(np.clip((homogeneity + energy) / 2.0, 0.0, 1.0)),
        }

    @staticmethod
    def calculate_lai_from_mask(
        mask: np.ndarray,
        image: np.ndarray,
        crop_type: str = "wheat",
        lai_min: float = 0.0,
        lai_max: float = 6.0,
        indices: dict | None = None,
        vegetation_score: np.ndarray | None = None,
    ) -> dict:
        indices = indices if indices is not None else CVEngine.calculate_vegetation_indices(image)
        vegetation_score = vegetation_score if vegetation_score is not None else CVEngine.build_vegetation_score(indices, image, crop_type=crop_type)
        binary_coverage = float(np.count_nonzero(mask) / mask.size)
        soft_coverage = float(np.mean(vegetation_score))
        coverage = float(np.clip((0.75 * binary_coverage) + (0.25 * soft_coverage), 0.0, 0.995))
        texture = CVEngine.extract_texture_features(image, mask)
        mask_features = CVEngine.extract_mask_features(mask)

        coefficients = {
            "wheat": {"k": 0.61, "omega_min": 0.82},
            "soy": {"k": 0.61, "omega_min": 0.80},
            "sunflower": {"k": 0.68, "omega_min": 0.84},
            "corn": {"k": 0.64, "omega_min": 0.86},
            "mixed": {"k": 0.60, "omega_min": 0.80},
            "vertical": {"k": 0.60, "omega_min": 0.78},
        }
        params = coefficients.get(crop_type, coefficients["mixed"])

        gap_fraction = float(np.clip(1.0 - coverage, 1e-3, 0.995))
        clumping_index = np.clip(
            1.02
            - (0.18 * (1.0 - mask_features["largest_component_ratio"]))
            - (0.004 * max(mask_features["component_count"] - 1, 0)),
            params["omega_min"],
            1.05,
        )
        effective_k = max(params["k"] * clumping_index, 0.18)
        lai_value = -np.log(gap_fraction) / effective_k

        # Light correction keeps very sparse and very dense masks stable.
        if coverage < 0.08:
            lai_value *= 0.92
        elif coverage > 0.75:
            lai_value *= 1.08

        lai_value += texture["texture_factor"] * 0.05

        lai_value = float(np.clip(lai_value, lai_min, lai_max)) if coverage > 0 else 0.0

        return {
            "lai_value": lai_value,
            "coverage": coverage,
            "binary_coverage": binary_coverage,
            "soft_coverage": soft_coverage,
            "gap_fraction": gap_fraction,
            "extinction_coefficient": effective_k,
            "clumping_index": float(clumping_index),
            **mask_features,
            **texture,
        }

    @staticmethod
    def estimate_confidence_v2(image: np.ndarray, mask: np.ndarray, lai_result: dict) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask_pixels = int(np.count_nonzero(mask))
        coverage = float(lai_result.get("coverage", 0.0))
        texture_factor = float(lai_result.get("texture_factor", 0.0))

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        contrast_score = float(np.clip(laplacian_var / 600.0, 0.0, 1.0))
        coverage_score = float(np.clip(coverage / 0.45, 0.0, 1.0))

        if mask_pixels:
            vegetation_values = gray[mask > 0]
            brightness_std = float(np.std(vegetation_values))
            brightness_score = float(np.clip(brightness_std / 35.0, 0.0, 1.0))
            mask_balance = float(np.clip(1.0 - abs(coverage - 0.45) / 0.45, 0.0, 1.0))
        else:
            brightness_score = 0.0
            mask_balance = 0.0

        edges = cv2.Canny(gray, 80, 160)
        edge_ratio = float(np.count_nonzero(edges) / edges.size)
        edge_score = float(np.clip(1.0 - (edge_ratio / 0.28), 0.0, 1.0))

        confidence = float(
            0.25 * contrast_score
            + 0.25 * coverage_score
            + 0.15 * brightness_score
            + 0.20 * texture_factor
            + 0.15 * edge_score
        )

        flags = {
            "low_contrast": contrast_score < 0.25,
            "sparse_coverage": coverage < 0.05,
            "excessive_edges": edge_ratio > 0.25,
            "poor_texture": texture_factor < 0.15,
        }

        if confidence >= 0.75:
            recommendation = "Result is reliable for field screening."
        elif flags["sparse_coverage"]:
            recommendation = "Too little vegetation in frame. Capture a denser canopy area."
        elif flags["low_contrast"]:
            recommendation = "Low image contrast. Re-shoot under softer and more even light."
        elif flags["excessive_edges"]:
            recommendation = "The image looks noisy or busy. Avoid shadows and background objects."
        elif flags["poor_texture"]:
            recommendation = "Leaf texture is weak. Move closer or improve focus."
        else:
            recommendation = "Quality is acceptable, but the image can be improved for a more stable LAI estimate."

        return {
            "confidence": confidence,
            "metrics": {
                "contrast_score": contrast_score,
                "coverage_score": coverage_score,
                "brightness_score": brightness_score,
                "mask_balance": mask_balance,
                "texture_score": texture_factor,
                "edge_score": edge_score,
            },
            "flags": flags,
            "recommendation": recommendation,
            "estimated_error": float(0.25 + (1.0 - confidence) * 0.85),
        }
