import base64
import io
import logging
import time
from urllib.parse import urlencode

import cv2
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError
from odoo import http
from odoo.http import request

_logger = logging.getLogger(__name__)
MAX_IMAGE_SIZE = 5 * 1024 * 1024
VALID_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}
VALID_METHODS = {"cv_ensemble", "sam"}
VALID_CROP_TYPES = {"wheat", "corn", "sunflower", "soy", "mixed", "vertical"}


class LAICalculatorController(http.Controller):
    def _normalize_uploaded_image(self, image_data: bytes) -> bytes:
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                image = image.convert("RGB")
                image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85, optimize=True)
                return buffer.getvalue()
        except (UnidentifiedImageError, OSError):
            # Some uploads are valid WEBP/other formats with wrong file extension
            # or Pillow is missing the corresponding codec. Fall back to OpenCV.
            np_buffer = np.frombuffer(image_data, dtype=np.uint8)
            decoded = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            if decoded is None:
                raise

            resized = decoded
            height, width = resized.shape[:2]
            max_dim = max(height, width)
            if max_dim > 1400:
                scale = 1400.0 / float(max_dim)
                resized = cv2.resize(
                    resized,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise ValueError("Failed to encode normalized JPEG.")
            return encoded.tobytes()

    def _validate_calibration_params(self, params):
        try:
            hue_center = float(params.get("custom_green_hue_center", 0.17))
            hue_width = float(params.get("custom_green_hue_width", 3.0))
            lai_min = float(params.get("custom_lai_min", 0.5))
            lai_max = float(params.get("custom_lai_max", 6.0))
        except (TypeError, ValueError):
            return False, "Invalid number format in calibration parameters."

        if not (0.0 <= hue_center <= 1.0):
            return False, "Green Hue Center must be between 0 and 1."
        if not (0.1 <= hue_width <= 10.0):
            return False, "Hue Sensitivity must be between 0.1 and 10."
        if not (0.0 <= lai_min <= 7.9):
            return False, "Min LAI must be between 0 and 7.9."
        if not (0.1 <= lai_max <= 8.0):
            return False, "Max LAI must be between 0.1 and 8."
        if lai_min >= lai_max:
            return False, "Min LAI must be less than Max LAI."

        return True, ""

    def _redirect_with_error(self, error: str, params: dict | None = None):
        query_params = {"error": error}
        if params:
            query_params.update({key: value for key, value in params.items() if key != "image" and value not in (None, "")})
        return request.redirect(f"/lai-calculator?{urlencode(query_params)}")

    @http.route("/lai-calculator", type="http", auth="public", website=True)
    def lai_calculator(self, **kw):
        return request.render(
            "lai_estimator.lai_calculator_page",
            {
                "use_custom_calibration": kw.get("use_custom_calibration") == "on",
                "custom_green_hue_center": kw.get("custom_green_hue_center", "0.17"),
                "custom_green_hue_width": kw.get("custom_green_hue_width", "3.0"),
                "custom_lai_min": kw.get("custom_lai_min", "0.5"),
                "custom_lai_max": kw.get("custom_lai_max", "6.0"),
                "segmentation_method": kw.get("segmentation_method", "cv_ensemble"),
            },
        )

    @http.route("/lai-calculate", type="http", auth="public", methods=["POST"], website=True, csrf=False)
    def lai_calculate(self, **kw):
        image_file = kw.get("image")
        if not image_file:
            _logger.warning("No image file in request")
            return self._redirect_with_error("NoImage")

        try:
            image_data = image_file.read()
        except Exception as exc:
            _logger.error("Failed to read image file: %s", exc)
            return self._redirect_with_error("ReadError")

        if not image_data:
            return self._redirect_with_error("EmptyImage")
        if len(image_data) > MAX_IMAGE_SIZE:
            _logger.warning("Image too large: %s bytes", len(image_data))
            return self._redirect_with_error("ImageTooLarge")

        try:
            image_data = self._normalize_uploaded_image(image_data)
        except Exception as exc:
            _logger.error("Failed to normalize uploaded image: %s", exc)
            return self._redirect_with_error("InvalidImage")

        filename = image_file.filename or f"upload_{int(time.time())}.bin"
        if "." in filename:
            ext = filename.rsplit(".", 1)[-1].lower()
            if ext not in VALID_EXTENSIONS:
                return self._redirect_with_error("InvalidFormat", {"ext": ext})

        crop_type = kw.get("crop_type", "wheat")
        if crop_type not in VALID_CROP_TYPES:
            crop_type = "wheat"

        segmentation_method = kw.get("segmentation_method", "cv_ensemble")
        if segmentation_method not in VALID_METHODS:
            segmentation_method = "cv_ensemble"

        use_custom = bool(kw.get("use_custom_calibration"))
        if use_custom:
            is_valid, error_msg = self._validate_calibration_params(kw)
            if not is_valid:
                return self._redirect_with_error(error_msg, kw)

        def safe_float(value, default):
            try:
                return float(value) if value not in (None, "") else default
            except (TypeError, ValueError):
                return default

        try:
            calc = request.env["lai.calculation"].sudo().create(
                {
                    "name": f"LAI-{request.env.user.name or 'Guest'}-{int(time.time())}",
                    "image": base64.b64encode(image_data).decode("ascii"),
                    "image_filename": f"{filename.rsplit('.', 1)[0]}.jpg" if "." in filename else f"{filename}.jpg",
                    "crop_type": crop_type,
                    "segmentation_method": segmentation_method,
                    "use_custom_calibration": use_custom,
                    "custom_green_hue_center": safe_float(kw.get("custom_green_hue_center"), 0.17),
                    "custom_green_hue_width": safe_float(kw.get("custom_green_hue_width"), 3.0),
                    "custom_lai_min": safe_float(kw.get("custom_lai_min"), 0.5),
                    "custom_lai_max": safe_float(kw.get("custom_lai_max"), 6.0),
                }
            )

            (
                avg_lai,
                heatmap_bytes,
                heatmap_filename,
                confidence,
                coverage,
                texture_hom,
                recommendation,
                status_info,
            ) = calc._process_image_and_calculate_lai(image_data, crop_type)

            calc.write(
                {
                    "lai_avg": avg_lai,
                    "lai_heatmap": base64.b64encode(heatmap_bytes).decode("ascii"),
                    "lai_heatmap_filename": heatmap_filename,
                    "confidence": confidence,
                    "coverage_percent": coverage,
                    "texture_homogeneity": texture_hom,
                    "recommendation_text": recommendation,
                    "crop_status": status_info.get("status", "unknown"),
                    "status_confidence": status_info.get("confidence", 0.0),
                }
            )

            return request.redirect(f"/lai-result/{calc.id}")
        except ValueError as exc:
            _logger.error("Validation error: %s", exc)
            return self._redirect_with_error("ValidationError", {"msg": str(exc)[:150]})
        except MemoryError:
            _logger.error("MemoryError during LAI processing")
            return self._redirect_with_error("ServerOverloaded")
        except Exception as exc:
            _logger.exception("Unexpected error in LAI calculation: %s", exc)
            return self._redirect_with_error("ProcessingFailed", {"msg": str(exc)[:150]})

    @http.route("/lai-result/<int:calc_id>", type="http", auth="public", website=True)
    def lai_result(self, calc_id, **kw):
        calc = request.env["lai.calculation"].sudo().browse(calc_id)
        if not calc.exists():
            return request.redirect("/lai-calculator?error=NotFound")

        date_iso = (calc.date_calculated.isoformat() + "Z") if calc.date_calculated else None
        return request.render(
            "lai_estimator.lai_result_page",
            {
                "calc": calc,
                "date_iso": date_iso,
            },
        )
