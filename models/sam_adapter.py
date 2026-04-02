import logging
import numpy as np
from pathlib import Path

_logger = logging.getLogger(__name__)


class SAMAdapter: 
    _predictor = None
    _initialized = False
    _model_type = 'vit_b'
    
    @classmethod
    def initialize(cls, checkpoint_path: str, device: str = 'cpu') -> bool:

        if cls._initialized and cls._predictor is not None:
            return True
            
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import torch
            
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            sam = sam_model_registry[cls._model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            sam.eval()
            
            cls._predictor = SamPredictor(sam)
            cls._device = device
            cls._initialized = True
            
            _logger.info(f"SAM {cls._model_type} loaded on {device}")
            return True
            
        except ImportError as e:
            _logger.error(f"Missing dependency: {e}. Install: pip install segment-anything")
            raise
        except Exception as e:
            _logger.exception(f"Failed to initialize SAM: {e}")
            cls._initialized = False
            raise
    
    @classmethod
    def segment_vegetation(cls, rgb_image: np.ndarray, prompt_box: list = None, min_area_ratio: float = 0.001) -> np.ndarray:

        if not cls._initialized or cls._predictor is None:
            raise RuntimeError("SAM not initialized. Call initialize() first.")
        
        import torch
        import cv2
        
        h, w = rgb_image.shape[:2]
        
        cls._predictor.set_image(rgb_image)
        
        if prompt_box is None:
            prompt_box = cls._find_green_region_hsv(rgb_image)
            if prompt_box is None:
                return np.zeros((h, w), dtype=np.uint8)
        
        with torch.no_grad():
            masks, scores, _ = cls._predictor.predict(box=np.array(prompt_box), multimask_output=True)
        
        best_idx = np.argmax(scores)
        mask = (masks[best_idx] * 255).astype(np.uint8)
        mask = cls._postprocess_mask(mask, min_area_ratio=min_area_ratio)
        
        return mask
    
    @classmethod
    def _find_green_region_hsv(cls, rgb_image: np.ndarray) -> list | None:
        import cv2
        
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        return [int(x), int(y), int(x + cw), int(y + ch)]
    
    @classmethod
    def _postprocess_mask(cls, mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
        import cv2
        
        h, w = mask.shape
        min_area = int(h * w * min_area_ratio)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask[labels == i] = 0
        
        return mask
    
    @classmethod
    def clear_cache(cls):
        if cls._predictor:
            cls._predictor.features = None
            cls._predictor.orig_h = None
            cls._predictor.orig_w = None
            cls._predictor.input_h = None
            cls._predictor.input_w = None
        _logger.debug("SAM cache cleared")
    
    @classmethod
    def reset(cls):
        cls._predictor = None
        cls._initialized = False
        import gc
        gc.collect()
        _logger.info("SAM adapter reset")

    @classmethod
    def aggressive_cleanup(cls):

        import gc
        
        cls.clear_cache()
        
        try:
            import torch
            if torch.cuda.is_available():
                _ = torch.cuda.device_count()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except (RuntimeError, AttributeError):
                    pass
        except (RuntimeError, ImportError, AttributeError):
            pass
        
        gc.collect()
        gc.collect()