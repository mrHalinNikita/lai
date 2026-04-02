import logging
import numpy as np
from PIL import Image as PILImage

_logger = logging.getLogger(__name__)


class CLIPAdapter:
    
    _model = None
    _processor = None
    _initialized = False
    _device = 'cpu'
    
    @classmethod
    def initialize(cls, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):

        if cls._initialized:
            return True
            
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            cls._processor = CLIPProcessor.from_pretrained(model_name)
            cls._model = CLIPModel.from_pretrained(model_name)
            cls._model.to(device).eval()
            cls._device = device
            cls._initialized = True
            
            _logger.info(f"CLIP {model_name} loaded on {device}")
            return True
            
        except ImportError:
            _logger.error("Missing 'transformers' or 'torch'. Run: pip install transformers torch")
            raise
        except Exception as e:
            _logger.exception(f"CLIP initialization failed: {e}")
            cls._initialized = False
            raise

    @classmethod
    def assess_crop_status(cls, rgb_image: np.ndarray, prompts: list = None) -> dict:

        if not cls._initialized:
            cls.initialize()
            
        if prompts is None:
            prompts = [
                "healthy green crop field with dense vegetation",
                "stressed yellowing plants showing drought or nutrient deficiency",
                "weed-infested agricultural field with mixed vegetation",
                "bare soil with very sparse or early-stage vegetation"
            ]
            
        try:
            import torch
            
            pil_img = PILImage.fromarray(rgb_image)
            inputs = cls._processor(
                text=prompts, 
                images=pil_img, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(cls._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = cls._model(**inputs)
            
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()
            best_idx = int(probs.argmax())
            
            return {
                "status": prompts[best_idx],
                "confidence": float(probs[best_idx]),
                "all_probabilities": dict(zip(prompts, probs.tolist()))
            }
            
        except Exception as e:
            _logger.warning(f"CLIP assessment failed: {e}")
            return {"status": "unknown", "confidence": 0.0, "all_probabilities": {}}

    @classmethod
    def cleanup(cls):

        cls._model = None
        cls._processor = None
        cls._initialized = False
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
            
        import gc
        gc.collect()
        _logger.debug("CLIP adapter cleaned up")