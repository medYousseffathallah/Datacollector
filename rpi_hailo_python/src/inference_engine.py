import time
import logging
import numpy as np
import cv2
import threading
from queue import Queue

logger = logging.getLogger("InferenceEngine")

try:
    import hailo_platform as hpf
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.warning("hailo_platform not found. Running in MOCK mode.")

class InferenceEngine:
    def __init__(self, config):
        self.config = config['inference']
        self.model_path = self.config['model_path']
        self.input_shape = tuple(self.config['input_shape']) # (640, 640)
        self.score_threshold = self.config.get('score_threshold', 0.5)
        self.running = False
        self.target = None
        self.network_group = None
        self.infer_pipeline = None
        self.hef = None
        
        if HAILO_AVAILABLE:
            self._init_hailo()
        else:
            self._init_mock()

    def _init_hailo(self):
        try:
            self.hef = hpf.HEF(self.model_path)
            
            # Configure Params
            self.params = hpf.VDevice.create_params()
            # self.params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN 
            
            self.target = hpf.VDevice(params=self.params)
            
            # Configure Network Group
            configure_params = hpf.ConfigureParams.create_from_hef(
                hef=self.hef, 
                interface=hpf.HailoStreamInterface.PCIe
            )
            self.network_groups = self.target.configure(self.hef, configure_params)
            self.network_group = self.network_groups[0]
            
            self.network_group_params = self.network_group.create_params()
            
            # Get Stream Infos
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_infos = self.hef.get_output_vstream_infos()
            
            # Create Params for VStreams
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
            HAILO_AVAILABLE = False
            self._init_mock()

    def _init_mock(self):
        logger.info("Initialized Mock Inference Engine.")

    def start(self):
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
        self.running = False
        if HAILO_AVAILABLE and self.target:
            if hasattr(self, 'pipeline') and self.pipeline:
                self.pipeline.__exit__(None, None, None)
            # Deactivate is handled by context managers usually, but here we did manual activation
            # self.network_group.deactivate() # Not always needed if using 'with' block, but here we persist
            self.target.release()

    def preprocess(self, frame):
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
        Returns: (masks, class_ids, scores)
        """
        if not self.running:
            return None

        processed_frame = self.preprocess(frame)
        if HAILO_AVAILABLE and self.target:
            input_data = {self.input_vstream_info.name: np.expand_dims(processed_frame, axis=0)}
            results = self.pipeline.infer(input_data)
            return self.post_process(results, frame.shape)
        else:
            return self.mock_inference(frame.shape)

    def post_process(self, results, original_shape):
        """
        Convert raw Hailo output to masks/boxes.
        This is highly dependent on the specific HEF output layers.
        For this template, we assume a simplified output structure or placeholder.
        """
        # Placeholder for real post-processing logic
        # You would typically:
        # 1. Decode boxes
        # 2. Decode masks (matrix mult with protos)
        # 3. NMS
        # For now, we return empty list to avoid crashing if user runs this without specific logic
        return [], [], []

    def mock_inference(self, shape):
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
