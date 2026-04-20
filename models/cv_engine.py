import cv2
import numpy as np


class CVEngine:
    WORKING_MAX_DIM = 1024
    HEATMAP_BLUR_SIZE = 31

    SCIENTIFIC_METHOD = "Scientific RGB v4"
    SCIENTIFIC_BASIS = (
        "Indirect optical LAI estimation from visible gap fraction using Beer-Lambert inversion "
        "and RGB vegetation segmentation based on green indices."
    )

    SCENE_PARAMETERS = {
        "wheat": {"k": 0.88, "omega_min": 0.84, "cover_bias": 0.28, "closure_sensitivity": 0.46},
        "corn": {"k": 0.84, "omega_min": 0.85, "cover_bias": 0.25, "closure_sensitivity": 0.40},
        "sunflower": {"k": 0.90, "omega_min": 0.86, "cover_bias": 0.30, "closure_sensitivity": 0.44},
        "soy": {"k": 0.86, "omega_min": 0.83, "cover_bias": 0.30, "closure_sensitivity": 0.48},
        "mixed": {"k": 0.92, "omega_min": 0.82, "cover_bias": 0.32, "closure_sensitivity": 0.52},
        "vertical": {"k": 1.02, "omega_min": 0.88, "cover_bias": 0.22, "closure_sensitivity": 0.55},
    }

    @staticmethod
    def _safe_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return default
        return float(np.percentile(flat, q))

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

        rgb = image.astype(np.float32)
        channel_means = np.maximum(np.mean(rgb, axis=(0, 1)), 1.0)
        target = float(np.mean(channel_means))
        gains = np.clip(target / channel_means, 0.75, 1.35)
        balanced = np.clip(rgb * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(balanced, (3, 3), 0)

    @staticmethod
    def detect_analysis_roi(image: np.ndarray) -> tuple[int, int, int, int] | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        red_low = cv2.inRange(hsv, np.array([0, 90, 70], dtype=np.uint8), np.array([14, 255, 255], dtype=np.uint8))
        red_high = cv2.inRange(hsv, np.array([166, 90, 70], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
        red_mask = cv2.bitwise_or(red_low, red_high)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        red_points = np.column_stack(np.where(red_mask > 0))
        if red_points.shape[0] < max(120, int(image.shape[0] * image.shape[1] * 0.002)):
            return None

        y_coords = red_points[:, 0]
        x_coords = red_points[:, 1]
        x1, x2 = int(x_coords.min()), int(x_coords.max())
        y1, y2 = int(y_coords.min()), int(y_coords.max())
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        if w < image.shape[1] * 0.18 or h < image.shape[0] * 0.18:
            return None
        if w > image.shape[1] * 0.92 or h > image.shape[0] * 0.92:
            return None

        border = max(6, int(min(w, h) * 0.05))
        border_mask = np.zeros_like(red_mask)
        border_mask[y1:y1 + border, x1:x2 + 1] = 255
        border_mask[y2 - border + 1:y2 + 1, x1:x2 + 1] = 255
        border_mask[y1:y2 + 1, x1:x1 + border] = 255
        border_mask[y1:y2 + 1, x2 - border + 1:x2 + 1] = 255

        border_hit_ratio = float(np.count_nonzero((red_mask > 0) & (border_mask > 0)) / max(np.count_nonzero(red_mask), 1))
        if border_hit_ratio < 0.72:
            return None

        inner_margin = max(8, int(min(w, h) * 0.06))
        ix1 = min(max(x1 + inner_margin, 0), image.shape[1] - 1)
        iy1 = min(max(y1 + inner_margin, 0), image.shape[0] - 1)
        ix2 = max(min(x2 - inner_margin, image.shape[1]), ix1 + 2)
        iy2 = max(min(y2 - inner_margin, image.shape[0]), iy1 + 2)
        return ix1, iy1, ix2, iy2

    @staticmethod
    def normalize_map(values: np.ndarray, lower_q: float = 5.0, upper_q: float = 95.0) -> np.ndarray:
        lower = float(np.percentile(values, lower_q))
        upper = float(np.percentile(values, upper_q))
        if upper <= lower:
            return np.zeros_like(values, dtype=np.float32)
        normalized = (values.astype(np.float32) - lower) / (upper - lower)
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def calculate_vegetation_indices(image: np.ndarray) -> dict:
        img = image.astype(np.float32) / 255.0
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        eps = 1e-6

        exg = (2.0 * g) - r - b
        exr = (1.4 * r) - g
        exgr = exg - exr
        ndi = (g - r) / (g + r + eps)
        vari = (g - r) / (g + r - b + eps)
        gli = ((2.0 * g) - r - b) / ((2.0 * g) + r + b + eps)
        cive = (0.441 * r) - (0.811 * g) + (0.385 * b) + (18.787 / 255.0)
        green_ratio = g / (r + g + b + eps)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue = hsv[:, :, 0] / 179.0
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        a_green = 1.0 - np.clip(lab[:, :, 1] / 255.0, 0.0, 1.0)
        b_yellow = np.clip(lab[:, :, 2] / 255.0, 0.0, 1.0)

        hue_center = 60.0 / 179.0
        hue_distance = np.minimum(np.abs(hue - hue_center), 1.0 - np.abs(hue - hue_center))
        hue_score = np.clip(1.0 - (hue_distance / 0.18), 0.0, 1.0)

        return {
            "r": r,
            "g": g,
            "b": b,
            "exg": exg,
            "exgr": exgr,
            "ndi": ndi,
            "vari": vari,
            "gli": gli,
            "cive": cive,
            "green_ratio": green_ratio,
            "hue_score": hue_score,
            "saturation": saturation,
            "value": value,
            "a_green": a_green,
            "b_yellow": b_yellow,
        }

    @staticmethod
    def assess_scene_conditions(image: np.ndarray, indices: dict) -> dict:
        del image
        value = indices["value"]
        saturation = indices["saturation"]

        shadow_ratio = float(np.mean(value < 0.22))
        highlight_ratio = float(np.mean(value > 0.92))
        low_sat_ratio = float(np.mean(saturation < 0.14))
        dynamic_range = float(np.percentile(value, 95) - np.percentile(value, 5))

        if shadow_ratio > 0.18 or (highlight_ratio > 0.08 and shadow_ratio > 0.08):
            illumination = "harsh"
        elif low_sat_ratio > 0.40 and dynamic_range < 0.35:
            illumination = "flat"
        else:
            illumination = "balanced"

        return {
            "illumination": illumination,
            "shadow_ratio": shadow_ratio,
            "highlight_ratio": highlight_ratio,
            "low_saturation_ratio": low_sat_ratio,
            "dynamic_range": dynamic_range,
        }

    @staticmethod
    def build_vegetation_score(indices: dict, image: np.ndarray, crop_type: str = "mixed") -> np.ndarray:
        exg_score = CVEngine.normalize_map(indices["exg"], 8.0, 96.0)
        exgr_score = CVEngine.normalize_map(indices["exgr"], 8.0, 96.0)
        ndi_score = CVEngine.normalize_map(indices["ndi"], 5.0, 95.0)
        vari_score = CVEngine.normalize_map(indices["vari"], 5.0, 95.0)
        gli_score = CVEngine.normalize_map(indices["gli"], 5.0, 95.0)
        cive_score = CVEngine.normalize_map(-indices["cive"], 5.0, 95.0)
        hue_score = indices["hue_score"].astype(np.float32)
        sat_score = indices["saturation"].astype(np.float32)
        green_ratio = CVEngine.normalize_map(indices["green_ratio"], 8.0, 96.0)
        lab_green = indices["a_green"].astype(np.float32)
        value_penalty = 1.0 - np.clip(np.abs(indices["value"] - 0.58) / 0.58, 0.0, 1.0)
        green_balance = np.clip((indices["g"] - np.maximum(indices["r"], indices["b"])) * 1.8 + 0.5, 0.0, 1.0)

        scene = CVEngine.assess_scene_conditions(image, indices)

        if crop_type == "vertical":
            score = (
                0.18 * exg_score
                + 0.12 * exgr_score
                + 0.12 * ndi_score
                + 0.10 * vari_score
                + 0.08 * gli_score
                + 0.12 * cive_score
                + 0.12 * hue_score
                + 0.08 * sat_score
                + 0.08 * green_ratio
                + 0.10 * lab_green
            )
        else:
            score = (
                0.20 * exg_score
                + 0.15 * exgr_score
                + 0.14 * ndi_score
                + 0.12 * vari_score
                + 0.10 * gli_score
                + 0.10 * cive_score
                + 0.07 * hue_score
                + 0.05 * sat_score
                + 0.04 * green_ratio
                + 0.03 * lab_green
            )

        if scene["illumination"] == "harsh":
            score = (0.88 * score) + (0.12 * value_penalty)
        elif scene["illumination"] == "flat":
            score = (0.92 * score) + (0.08 * hue_score)

        score = (0.90 * score) + (0.10 * green_balance)
        return np.clip(score.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def build_support_masks(indices: dict, image: np.ndarray, hue_center: float = 0.17, hue_width: float = 3.0) -> dict:
        hsv_mask = CVEngine.segment_green_by_hsv(image, hue_center=hue_center, hue_width=hue_width) > 0
        green_dominant = (indices["g"] > indices["r"] * 0.90) & (indices["g"] > indices["b"] * 0.86)
        exg_positive = indices["exg"] > max(0.0, float(np.percentile(indices["exg"], 52)))
        exgr_positive = indices["exgr"] > float(np.percentile(indices["exgr"], 55))
        ndi_positive = indices["ndi"] > max(0.02, float(np.percentile(indices["ndi"], 55)))
        hue_green = indices["hue_score"] > 0.34
        sat_ok = indices["saturation"] > 0.10
        brightness_ok = indices["value"] > 0.12

        broad_support = (hsv_mask | green_dominant | exg_positive | exgr_positive) & hue_green & sat_ok & brightness_ok
        strong_support = ((green_dominant & exg_positive) | (hsv_mask & ndi_positive) | (exg_positive & exgr_positive)) & brightness_ok

        return {
            "hsv_mask": hsv_mask,
            "green_dominant": green_dominant,
            "exg_positive": exg_positive,
            "exgr_positive": exgr_positive,
            "ndi_positive": ndi_positive,
            "broad_support": broad_support,
            "strong_support": strong_support,
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

        lower = np.array([max(0, hue - width), 30, 20], dtype=np.uint8)
        upper = np.array([min(179, hue + width), 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    @staticmethod
    def refine_mask_adaptive(mask: np.ndarray, image: np.ndarray, crop_type: str = "mixed") -> np.ndarray:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)

        min_area_ratio = 0.0012 if crop_type == "vertical" else 0.0007
        min_area = max(24, int(image.shape[0] * image.shape[1] * min_area_ratio))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
        cleaned = np.zeros_like(refined)

        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == idx] = 255

        return cleaned

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
        del weights
        score = vegetation_score if vegetation_score is not None else CVEngine.build_vegetation_score(indices, image, crop_type=crop_type)
        support = CVEngine.build_support_masks(indices, image, hue_center=hue_center, hue_width=hue_width)
        candidate = support["broad_support"] | support["strong_support"]

        candidate_scores = score[candidate]
        if candidate_scores.size < 128:
            candidate_scores = score.reshape(-1)

        score_u8 = np.clip(candidate_scores * 255.0, 0, 255).astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        base_threshold = float(otsu_threshold) / 255.0
        threshold = max(base_threshold, 0.28 if crop_type == "vertical" else 0.18)
        soft_threshold = max(0.10, threshold - 0.08)

        core = (score >= threshold) | ((score >= soft_threshold + 0.03) & support["strong_support"])
        secondary = (
            ((score >= soft_threshold) & support["broad_support"])
            | ((score >= soft_threshold - 0.02) & support["exg_positive"] & support["ndi_positive"])
        )
        dilated_core = cv2.dilate(np.where(core, 255, 0).astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
        mask = core | (secondary & dilated_core)

        mask = np.where(mask, 255, 0).astype(np.uint8)
        return CVEngine.refine_mask_adaptive(mask, image, crop_type=crop_type)

    @staticmethod
    def extract_mask_features(mask: np.ndarray) -> dict:
        if not np.any(mask):
            return {
                "component_count": 0,
                "largest_component_ratio": 0.0,
                "coverage": 0.0,
                "fragmentation": 1.0,
            }

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
        mask_area = float(np.count_nonzero(mask))
        largest_ratio = float(component_areas.max() / mask_area) if component_areas.size else 0.0
        fragmentation = float(np.clip(component_areas.size / 18.0, 0.0, 1.0))

        return {
            "component_count": int(component_areas.size),
            "largest_component_ratio": largest_ratio,
            "coverage": float(mask_area / mask.size),
            "fragmentation": fragmentation,
        }

    @staticmethod
    def extract_texture_features(image: np.ndarray, mask: np.ndarray) -> dict:
        if not np.any(mask):
            return {
                "texture_contrast": 0.0,
                "texture_homogeneity": 0.0,
                "texture_energy": 0.0,
                "texture_factor": 0.0,
            }

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        values = gray[mask > 0]

        if values.size == 0:
            return {
                "texture_contrast": 0.0,
                "texture_homogeneity": 0.0,
                "texture_energy": 0.0,
                "texture_factor": 0.0,
            }

        contrast = float(np.std(values))
        p10 = float(np.percentile(values, 10))
        p90 = float(np.percentile(values, 90))
        homogeneity = float(np.clip(1.0 - ((p90 - p10) / 255.0), 0.0, 1.0))
        energy = float(np.clip(np.mean(mask > 0), 0.0, 1.0))
        texture_factor = float(np.clip((homogeneity + (1.0 - np.clip(contrast / 90.0, 0.0, 1.0))) / 2.0, 0.0, 1.0))

        return {
            "texture_contrast": contrast,
            "texture_homogeneity": homogeneity,
            "texture_energy": energy,
            "texture_factor": texture_factor,
        }

    @staticmethod
    def build_density_map(mask: np.ndarray, vegetation_score: np.ndarray) -> np.ndarray:
        mask_float = (mask > 0).astype(np.float32)
        local_cover = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=7, sigmaY=7)
        local_score = cv2.GaussianBlur(vegetation_score.astype(np.float32), (0, 0), sigmaX=5, sigmaY=5)
        density = np.clip((0.72 * local_cover) + (0.28 * local_score), 0.0, 1.0)
        return density.astype(np.float32)

    @staticmethod
    def build_local_lai_map(mask: np.ndarray, vegetation_score: np.ndarray, effective_k: float, lai_cap: float) -> np.ndarray:
        mask_float = (mask > 0).astype(np.float32)
        local_cover = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=17, sigmaY=17)
        local_support = cv2.GaussianBlur(vegetation_score.astype(np.float32), (0, 0), sigmaX=9, sigmaY=9)
        local_density = np.clip((0.84 * local_cover) + (0.16 * local_support), 0.0, 0.985)
        local_gap = np.clip(1.0 - local_density, 0.02, 0.995)
        local_lai = -np.log(local_gap) / max(effective_k, 0.35)
        local_lai = np.clip(local_lai, 0.0, max(lai_cap, 0.25))
        return local_lai.astype(np.float32)

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

        params = CVEngine.SCENE_PARAMETERS.get(crop_type, CVEngine.SCENE_PARAMETERS["mixed"])
        mask_features = CVEngine.extract_mask_features(mask)
        texture = CVEngine.extract_texture_features(image, mask)
        scene = CVEngine.assess_scene_conditions(image, indices)

        binary_coverage = float(np.count_nonzero(mask) / mask.size)
        soft_coverage = float(np.mean(vegetation_score))
        semi_visible_bonus = max(soft_coverage - binary_coverage, 0.0)
        apparent_cover = float(
            np.clip(
                ((1.0 - params["cover_bias"]) * binary_coverage)
                + (params["cover_bias"] * soft_coverage)
                + (0.12 * semi_visible_bonus),
                0.0,
                0.985,
            )
        )
        gap_fraction = float(np.clip(1.0 - apparent_cover, 0.015, 0.995))

        clumping_index = float(
            np.clip(
                1.01
                - (0.10 * mask_features["fragmentation"])
                - (0.07 * (1.0 - mask_features["largest_component_ratio"]))
                + (0.04 * texture["texture_homogeneity"]),
                params["omega_min"],
                1.02,
            )
        )
        dynamic_k = params["k"] * (1.04 - (params["closure_sensitivity"] * apparent_cover * 0.55))
        effective_k = float(np.clip(dynamic_k * clumping_index, 0.42, 1.08))

        lai_value = float(-np.log(gap_fraction) / effective_k)
        if binary_coverage < 0.03:
            lai_value *= 0.85
        if crop_type == "vertical":
            lai_value *= 0.98

        lai_value = float(np.clip(lai_value, lai_min, lai_max)) if apparent_cover > 0 else 0.0
        density_map = CVEngine.build_density_map(mask, vegetation_score)
        local_lai_map = CVEngine.build_local_lai_map(mask, vegetation_score, effective_k, lai_max)
        lai_pixels = local_lai_map[local_lai_map > 0.02]

        return {
            "lai_value": lai_value,
            "coverage": binary_coverage,
            "apparent_cover": apparent_cover,
            "binary_coverage": binary_coverage,
            "soft_coverage": soft_coverage,
            "green_fraction": soft_coverage,
            "gap_fraction": gap_fraction,
            "extinction_coefficient": effective_k,
            "clumping_index": clumping_index,
            "illumination_class": scene["illumination"],
            "density_map": density_map,
            "local_lai_map": local_lai_map,
            "local_lai_mean": float(np.mean(lai_pixels)) if lai_pixels.size else 0.0,
            "local_lai_std": float(np.std(lai_pixels)) if lai_pixels.size else 0.0,
            "local_lai_p10": CVEngine._safe_percentile(lai_pixels, 10.0),
            "local_lai_p50": CVEngine._safe_percentile(lai_pixels, 50.0),
            "local_lai_p90": CVEngine._safe_percentile(lai_pixels, 90.0),
            "method_version": CVEngine.SCIENTIFIC_METHOD,
            "method_basis": CVEngine.SCIENTIFIC_BASIS,
            **mask_features,
            **texture,
        }

    @staticmethod
    def estimate_confidence_v2(image: np.ndarray, mask: np.ndarray, lai_result: dict) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        blur_score = float(np.clip(laplacian_var / 500.0, 0.0, 1.0))

        shadow_ratio = float(np.mean(gray < 35))
        highlight_ratio = float(np.mean(gray > 245))
        exposure_score = float(np.clip(1.0 - ((shadow_ratio + highlight_ratio) / 0.38), 0.0, 1.0))

        binary_coverage = float(lai_result.get("binary_coverage", 0.0))
        soft_coverage = float(lai_result.get("soft_coverage", 0.0))
        coverage_agreement = float(np.clip(1.0 - abs(binary_coverage - soft_coverage) / 0.35, 0.0, 1.0))
        texture_score = float(np.clip(lai_result.get("texture_factor", 0.0), 0.0, 1.0))

        coverage = float(lai_result.get("apparent_cover", lai_result.get("coverage", 0.0)))
        if 0.08 <= coverage <= 0.82:
            coverage_score = 1.0
        elif coverage < 0.08:
            coverage_score = float(np.clip(coverage / 0.08, 0.0, 1.0))
        else:
            coverage_score = float(np.clip((1.0 - coverage) / 0.18, 0.0, 1.0))

        confidence = float(
            0.28 * blur_score
            + 0.24 * exposure_score
            + 0.24 * coverage_agreement
            + 0.14 * coverage_score
            + 0.10 * texture_score
        )

        flags = {
            "blurry": blur_score < 0.25,
            "bad_exposure": exposure_score < 0.35,
            "weak_segmentation_agreement": coverage_agreement < 0.35,
            "sparse_coverage": coverage < 0.04,
            "high_saturation_risk": coverage > 0.90,
        }

        if confidence >= 0.78:
            recommendation = "Оценка LAI достаточно стабильна для полевого скрининга и сравнительных наблюдений."
        elif flags["weak_segmentation_agreement"]:
            recommendation = "Сегментация получилась нестабильной. Переснимите участок с более чистым отделением фона и более ровным кадрированием."
        elif flags["bad_exposure"]:
            recommendation = "Освещение неравномерное. Избегайте жестких теней и пересвеченных участков."
        elif flags["blurry"]:
            recommendation = "Снимок недостаточно резкий. Подойдите ближе и переснимите с лучшей фокусировкой."
        elif flags["sparse_coverage"]:
            recommendation = "Растительность занимает слишком малую часть кадра для уверенной оценки LAI."
        elif flags["high_saturation_risk"]:
            recommendation = "Покров очень плотный. Возможна недооценка LAI из-за оптического насыщения."
        else:
            recommendation = "Результат можно использовать, но более перпендикулярный и равномерно освещенный снимок повысит надежность."

        return {
            "confidence": confidence,
            "metrics": {
                "blur_score": blur_score,
                "exposure_score": exposure_score,
                "coverage_agreement": coverage_agreement,
                "coverage_score": coverage_score,
                "texture_score": texture_score,
            },
            "flags": flags,
            "recommendation": recommendation,
            "estimated_error": float(0.20 + (1.0 - confidence) * 0.90),
        }
