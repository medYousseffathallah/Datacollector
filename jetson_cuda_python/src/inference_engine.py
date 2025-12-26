import logging
import cv2
import numpy as np
import torch

logger = logging.getLogger("InferenceEngineJetson")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not found. Running in MOCK mode.")

class InferenceEngineJetson:
    def __init__(self, config):
        self.config = config['inference']
        self.model_path = self.config['model_path'] # Can be .pt or .engine
        self.score_threshold = self.config.get('score_threshold', 0.5)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        
        if ULTRALYTICS_AVAILABLE:
            self._init_model()
        else:
            self._init_mock()

    def _init_model(self):
        try:
            logger.info(f"Loading YOLO model from {self.model_path} on {self.device}...")
            self.model = YOLO(self.model_path)
            # Warmup
            self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            logger.info("Model loaded and warmed up.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _init_mock(self):
        logger.info("Initialized Mock Inference Engine (Jetson).")

    def start(self):
        # Ultralytics models are loaded in init, nothing specific to start
        pass

    def stop(self):
        # No specific cleanup needed for PyTorch model
        pass

    def infer(self, frame):
        """
        Run inference on a single frame.
        Returns: (masks, class_ids, scores)
        """
        if self.model is None:
            return self.mock_inference(frame.shape)

        # Run inference
        # stream=True for performance, verbose=False to reduce logs
        results = self.model(frame, conf=self.score_threshold, verbose=False, device=self.device)
        
        if not results:
            return [], [], []

        result = results[0] # Single frame
        
        if result.masks is None:
            return [], [], []

        # Extract data
        # masks.data is a torch tensor (N, H, W)
        # We need to convert to list of numpy arrays
        # Resize masks to original image size if they aren't already
        
        masks_tensor = result.masks.data
        boxes = result.boxes
        
        masks = []
        class_ids = []
        scores = []

        if masks_tensor is not None:
            # Resize masks to original frame size if needed
            # Ultralytics usually returns masks in input_shape (e.g. 640x640)
            # We need to resize them to frame.shape
            
            # Convert to CPU numpy
            masks_np = masks_tensor.cpu().numpy()
            
            for i, mask in enumerate(masks_np):
                # Resize to original frame size
                # Note: result.orig_shape is (h, w)
                mask_resized = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]))
                
                # Binarize
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                masks.append(mask_binary)
                class_ids.append(int(boxes.cls[i].item()))
                scores.append(float(boxes.conf[i].item()))

        return masks, class_ids, scores

    def mock_inference(self, shape):
        # Same mock logic as original
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w//2, h//2), h//4, 1, -1)
        return [mask], [0], [0.95]
