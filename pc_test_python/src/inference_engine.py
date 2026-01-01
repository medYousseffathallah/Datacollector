import time
import logging
import numpy as np
import cv2
import threading
from queue import Queue

from .utils import setup_logger

logger = setup_logger("InferenceEngine")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    
# Log status after checking all backends
if not ULTRALYTICS_AVAILABLE:
    logger.warning("ultralytics not found. System will run in MOCK mode.")

class InferenceEngine:
    """
    Wrapper for Ultralytics inference engine (PC Mode).
    """
    def __init__(self, config):
        """
        Initialize the inference engine.
        Args:
            config: Inference configuration dictionary.
        """
        self.config = config['inference']
        self.model_path = self.config['model_path']
        self.input_shape = tuple(self.config['input_shape']) # (640, 640)
        self.score_threshold = self.config.get('score_threshold', 0.5)
        self.running = False
        self.yolo_model = None
        
        # Initialize Ultralytics or Mock
        if ULTRALYTICS_AVAILABLE and self.model_path.endswith('.pt'):
             self._init_ultralytics()
        else:
            self._init_mock()

    def _init_ultralytics(self):
        """
        Initialize Ultralytics YOLO for PC testing.
        """
        try:
            logger.info(f"Loading Ultralytics model: {self.model_path}")
            self.yolo_model = YOLO(self.model_path)
            logger.info("Ultralytics model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Ultralytics model: {e}")
            self._init_mock()

    def _init_mock(self):
        """
        Initialize Mock engine for testing without hardware.
        """
        logger.info("Initialized Mock Inference Engine.")

    def start(self):
        """
        Activate the inference pipeline.
        """
        self.running = True

    def stop(self):
        """
        Stop the pipeline and release resources.
        """
        self.running = False

    def infer(self, frame):
        """
        Run inference on a single frame.
        Args:
            frame: Input image.
        Returns: 
            Tuple (masks, class_ids, scores)
        """
        if not self.running:
            return None
        
        if self.yolo_model:
            return self.infer_ultralytics(frame)
        else:
            return self.mock_inference(frame.shape)

    def infer_ultralytics(self, frame):
        """
        Run inference using Ultralytics YOLO (PC mode).
        """
        results = self.yolo_model(frame, verbose=False, conf=self.score_threshold)
        
        masks = []
        class_ids = []
        scores = []
        
        for r in results:
            if r.masks:
                # Segmentation results
                # r.masks.data contains the masks (N, H, W)
                masks_data = r.masks.data.cpu().numpy()
                
                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get mask for this detection
                    mask_raw = masks_data[i]
                    
                    # Ensure it is uint8 [0, 255]
                    # Ultralytics masks are often float [0,1], convert to uint8
                    if mask_raw.dtype != np.uint8:
                         mask_uint8 = (mask_raw * 255).astype(np.uint8)
                    else:
                         mask_uint8 = mask_raw
                    
                    masks.append(mask_uint8)
                    class_ids.append(cls_id)
                    scores.append(conf)
            
            elif r.boxes:
                # Detection results (no masks) -> Fallback to Box-as-Mask
                h, w = frame.shape[:2]
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Clip to image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # Create binary mask from box
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    
                    masks.append(mask)
                    class_ids.append(cls_id)
                    scores.append(conf)

        return masks, class_ids, scores

    def post_process(self, results, original_shape):
        """
        Convert raw Hailo output to masks/boxes.
        This is highly dependent on the specific HEF output layers.
        For this template, we assume a simplified output structure or placeholder.
        """
        # WARNING: This is a placeholder. 
        # Real Hailo models return raw tensors that need:
        # 1. Anchor decoding (if YOLO)
        # 2. Sigmoid/Softmax activation
        # 3. Non-Maximum Suppression (NMS)
        # 4. Mask prototype multiplication (if Segmentation)
        
        # If you are using a standard Hailo Model Zoo model, 
        # you should use the `hailo_model_zoo` post-processing utilities 
        # or `hailo_rpi5_examples` post-processing code.
        
        # logger.warning("Post-processing not implemented for this model. Returning empty results.")
        return [], [], []

    def mock_inference(self, shape):
        """
        Generate dummy detection data for testing.
        """
        # Generate a fake person detection
        h, w = shape[:2]
        
        # Fake Mask: a circle in the middle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w//2, h//2), h//4, 1, -1)
        
        masks = [mask]
        class_ids = [0] # 'person'
        scores = [0.95]
        
        # Simulate processing time (Hailo is fast, but let's say 30ms)
        time.sleep(0.03)
        
        return masks, class_ids, scores
