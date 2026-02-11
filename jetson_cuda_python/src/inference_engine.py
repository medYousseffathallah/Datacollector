import logging
import cv2
import numpy as np
import torch
import os

logger = logging.getLogger("InferenceEngineJetson")

try:
    # Ultralytics provides the YOLO engine for CUDA/TensorRT
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not found. Running in MOCK mode.")

class InferenceEngineJetson:
    """
    Inference Engine using Ultralytics YOLO.
    Supports PyTorch (.pt) and TensorRT (.engine) models on CUDA.
    """
    def __init__(self, config):
        """
        Initialize the inference engine.
        Args:
            config: Inference configuration dictionary.
        """
        self.config = config['inference']
        self.model_path = self.config['model_path']
        self.score_threshold = self.config.get('score_threshold', 0.5)
        self.auto_export_engine = bool(self.config.get('auto_export_engine', False))
        self.export_half = bool(self.config.get('export_half', True))
        self.export_device = self.config.get('export_device', 0)
        self.force_mock = bool(self.config.get('force_mock', False))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        
        logger.info(f"Inference Engine initializing on device: {self.device}")
        
        if self.force_mock:
            self._init_mock()
        elif ULTRALYTICS_AVAILABLE:
            self._init_model()
        else:
            self._init_mock()

    def _init_model(self):
        """
        Load the YOLO model and perform warmup.
        """
        try:
            self._maybe_export_engine()
            if self.model_path.endswith('.engine') and not os.path.exists(self.model_path):
                logger.warning(f"TensorRT engine file not found: {self.model_path}. Falling back to .pt if possible or error.")
            logger.info(f"Loading YOLO model from {self.model_path} on {self.device}...")
            self.model = YOLO(self.model_path)
            
            # Check if using TensorRT
            if self.model_path.endswith('.engine'):
                logger.info("Using TensorRT Engine for maximum performance.")
            
            # Warmup
            logger.info("Warming up model...")
            self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            logger.info("Model loaded and warmed up.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to MOCK mode due to model load failure.")
            self.model = None

    def _maybe_export_engine(self):
        if not self.auto_export_engine:
            return
        if not torch.cuda.is_available():
            return
        if not self.model_path.endswith('.pt'):
            return
        engine_path = os.path.splitext(self.model_path)[0] + '.engine'
        if os.path.exists(engine_path):
            self.model_path = engine_path
            return
        try:
            export_model = YOLO(self.model_path)
            export_model.export(format='engine', half=self.export_half, device=self.export_device)
            if os.path.exists(engine_path):
                self.model_path = engine_path
        except Exception as e:
            logger.warning(f"Engine export failed: {e}")

    def _init_mock(self):
        """
        Initialize Mock engine if dependencies are missing.
        """
        logger.info("Initialized Mock Inference Engine (Jetson).")

    def start(self):
        """
        No specific start required for PyTorch/Ultralytics models.
        """
        pass

    def stop(self):
        """
        No specific cleanup required.
        """
        pass

    def infer(self, frame):
        """
        Run inference on a single frame.
        Args:
            frame: Input image.
        Returns: 
            Tuple (masks, class_ids, scores)
        """
        if self.model is None:
            return self.mock_inference(frame.shape)

        # Run inference
        # stream=True for performance, verbose=False to reduce logs
        # device=self.device ensures we use CUDA if available
        try:
            results = self.model(frame, conf=self.score_threshold, verbose=False, device=self.device)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return [], [], []
        
        if not results:
            return [], [], []

        result = results[0] # Single frame
        
        # Initialize lists
        masks = []
        class_ids = []
        scores = []
        
        # Check if we have detections
        if result.boxes is None:
             return [], [], []
             
        boxes = result.boxes
        
        # If segmentation masks are available
        if result.masks is not None:
            masks_tensor = result.masks.data
            
            if masks_tensor is not None:
                # Ultralytics masks are usually (N, H, W) where H,W are input_shape (e.g. 640x640)
                # We need to map them back to original image size
                
                # Process each detection
                for i in range(len(boxes)):
                    # Get box info
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    
                    # Get the mask for this object
                    # Convert to CPU numpy
                    mask_raw = masks_tensor[i].cpu().numpy()
                    
                    # Resize to original frame size
                    # result.orig_shape is (h, w)
                    mask_resized = cv2.resize(mask_raw, (result.orig_shape[1], result.orig_shape[0]))
                    
                    # Binarize (Ultralytics masks are float 0-1)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    masks.append(mask_binary)
                    class_ids.append(cls_id)
                    scores.append(conf)
                    
        else:
            # Fallback for Object Detection models (no masks)
            # Create box masks
             for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                
                # Create mask from box
                h, w = result.orig_shape
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                masks.append(mask)
                class_ids.append(cls_id)
                scores.append(conf)

        return masks, class_ids, scores

    def mock_inference(self, shape):
        """
        Generate dummy detection data for testing.
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w//2, h//2), h//4, 1, -1)
        return [mask], [0], [0.95]
