import logging
import os

import numpy as np
from PIL import Image as PILImage

_logger = logging.getLogger(__name__)


class CLIPAdapter:
    _model = None
    _processor = None
    _initialized = False
    _device = "cpu"
    _enabled = os.environ.get("LAI_ENABLE_CLIP", "0") == "1"

    @classmethod
    def is_available(cls) -> bool:
        return cls._enabled

    @classmethod
    def initialize(cls, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        if not cls._enabled:
            return False

        if cls._initialized and cls._model is not None and cls._processor is not None:
            return True

        try:
            from transformers import CLIPModel, CLIPProcessor

            cls._processor = CLIPProcessor.from_pretrained(model_name)
            cls._model = CLIPModel.from_pretrained(model_name)
            cls._model.to(device).eval()
            cls._device = device
            cls._initialized = True
            _logger.info("CLIP %s loaded on %s", model_name, device)
            return True
        except ImportError:
            _logger.warning("CLIP dependencies are not available. Continuing without crop status assessment.")
            return False
        except Exception as exc:
            _logger.exception("CLIP initialization failed: %s", exc)
            cls._initialized = False
            return False

    @classmethod
    def assess_crop_status(cls, rgb_image: np.ndarray, prompts: list | None = None) -> dict:
        if not cls._enabled:
            return {"status": "unknown", "confidence": 0.0, "all_probabilities": {}}

        if not cls._initialized and not cls.initialize():
            return {"status": "unknown", "confidence": 0.0, "all_probabilities": {}}

        if prompts is None:
            prompts = [
                "healthy green crop field with dense vegetation",
                "stressed yellowing plants showing drought or nutrient deficiency",
                "weed-infested agricultural field with mixed vegetation",
                "bare soil with very sparse or early-stage vegetation",
            ]

        try:
            import torch

            pil_img = PILImage.fromarray(rgb_image)
            inputs = cls._processor(text=prompts, images=pil_img, return_tensors="pt", padding=True)
            inputs = {key: value.to(cls._device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = cls._model(**inputs)

            probabilities = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().numpy()
            best_idx = int(probabilities.argmax())
            return {
                "status": prompts[best_idx],
                "confidence": float(probabilities[best_idx]),
                "all_probabilities": dict(zip(prompts, probabilities.tolist())),
            }
        except Exception as exc:
            _logger.warning("CLIP assessment failed: %s", exc)
            return {"status": "unknown", "confidence": 0.0, "all_probabilities": {}}

    @classmethod
    def cleanup(cls):
        # Keep the model cached for future requests; explicit cleanup is optional.
        return None
