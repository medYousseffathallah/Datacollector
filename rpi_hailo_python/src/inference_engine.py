import time
import logging
import numpy as np
import cv2
import threading
from queue import Queue

from .utils import setup_logger

logger = setup_logger("InferenceEngine")

try:
    # Import Hailo Platform SDK
    import hailo_platform as hpf
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    # Don't log warning yet, wait to see if fallback is available

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    
# Log status after checking all backends
if not HAILO_AVAILABLE:
    if ULTRALYTICS_AVAILABLE:
        logger.info("hailo_platform not found. Will attempt to use Ultralytics fallback.")
    else:
        logger.warning("hailo_platform AND ultralytics not found. System will run in MOCK mode.")

class InferenceEngine:
    """
    Wrapper for HailoRT inference engine.
    Handles model loading, VStream configuration, and inference pipeline.
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
        self.target = None
        self.network_group = None
        self.infer_pipeline = None
        self.hef = None
        self.yolo_model = None
        
        # Initialize Hailo hardware if available, else use Mock/PC fallback
        if HAILO_AVAILABLE:
            self._init_hailo()
        elif ULTRALYTICS_AVAILABLE and self.model_path.endswith('.pt'):
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

    def _init_hailo(self):
        """
        Initialize HailoRT VDevice, HEF, and Network Groups.
        """
        try:
            # Load HEF model file
            self.hef = hpf.HEF(self.model_path)
            
            # Configure VDevice Parameters
            self.params = hpf.VDevice.create_params()
            # self.params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN 
            
            # Create VDevice (Access to PCIe device)
            self.target = hpf.VDevice(params=self.params)
            
            # Configure Network Group from HEF
            configure_params = hpf.ConfigureParams.create_from_hef(
                hef=self.hef, 
                interface=hpf.HailoStreamInterface.PCIe
            )
            self.network_groups = self.target.configure(self.hef, configure_params)
            self.network_group = self.network_groups[0]
            
            self.network_group_params = self.network_group.create_params()
            
            # Get Input/Output VStream Information
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_infos = self.hef.get_output_vstream_infos()
            
            # Create Parameters for VStreams (Quantized=False for float output)
            self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
            )
            self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
            )
            
            logger.info(f"Hailo Inference Engine initialized with model: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            self.target = None
            global HAILO_AVAILABLE
            HAILO_AVAILABLE = False
            self._init_mock()

    def _init_mock(self):
        """
        Initialize Mock engine for testing without hardware.
        """
        logger.info("Initialized Mock Inference Engine.")

    def start(self):
        """
        Activate the network group and start the inference pipeline.
        """
        self.running = True
        if HAILO_AVAILABLE and self.target:
            self.network_group.activate(self.network_group_params)
            # Create pipeline context
            self.pipeline = hpf.InferVStreams(
                self.network_group, 
                self.input_vstreams_params, 
                self.output_vstreams_params
            )
            self.pipeline.__enter__()

    def stop(self):
        """
        Stop the pipeline and release resources.
        """
        self.running = False
        if HAILO_AVAILABLE and self.target:
            if hasattr(self, 'pipeline') and self.pipeline:
                self.pipeline.__exit__(None, None, None)
            # Deactivate is handled by context managers usually, but here we did manual activation
            # self.network_group.deactivate() # Not always needed if using 'with' block, but here we persist
            self.target.release()

    def preprocess(self, frame):
        """
        Preprocess the input frame for the model.
        Args:
            frame: Raw input image (numpy array).
        Returns:
            Preprocessed frame (resized, normalized, float32).
        """
        # Resize and Normalize
        resized = cv2.resize(frame, self.input_shape)
        # Assuming model expects float32 normalized 0-1 or 0-255 depending on HEF
        # Usually Hailo HEF expects uint8 if quantized=True in VStream, but we set quantized=False, FLOAT32
        # So we likely need to normalize to 0-1 if the model was trained that way, 
        # BUT often Hailo Input conversion handles this.
        # Safe bet: pass float32 0-255 or 0-1. Let's assume 0-255 float.
        return resized.astype(np.float32)

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

        processed_frame = self.preprocess(frame)
        
        if HAILO_AVAILABLE and self.target:
            # Prepare input dictionary mapping stream name to data
            input_data = {self.input_vstream_info.name: np.expand_dims(processed_frame, axis=0)}
            # Run inference
            results = self.pipeline.infer(input_data)
            return self.post_process_hailo(results, frame.shape)
        elif self.yolo_model:
            return self.infer_ultralytics(frame)
        else:
            return self.mock_inference(frame.shape)

    def post_process_hailo(self, results, original_shape):
        """
        Process raw Hailo output tensors into masks, class_ids, and scores.
        NOTE: This is a placeholder structure. Actual implementation depends heavily
        on the specific output layers of the compiled HEF.
        
        Typically Hailo TAPPAS or hailo_model_zoo provides parsers.
        For raw YOLOv8 output, we usually get concatenated outputs or 3 separate heads.
        
        This simple implementation assumes the model was compiled with NMS on-chip
        or standard YOLO output structure.
        """
        # TODO: Implement specific decoding logic based on HEF output layers.
        # For now, return empty lists to prevent crash if run on hardware without full parser.
        # In a real scenario, you would parse 'results' which is a dict of {output_layer_name: numpy_array}
        
        # Example pseudo-code for a model compiled with meta-arch (NMS included):
        # detections = results['detections'] # shape [N, 6] (x1, y1, x2, y2, score, class)
        
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
