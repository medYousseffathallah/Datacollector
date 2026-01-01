import cv2
import time
import threading
import numpy as np
import logging
from queue import Queue, Empty

logger = logging.getLogger("CameraManager")

class MotionDetector:
    """
    Simple motion detector using frame differencing.
    """
    def __init__(self, threshold=25, min_area=500):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None

    def detect(self, frame):
        """
        Check if motion is detected in the frame compared to the previous frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return True # Always return True for the first frame

        # Compute difference
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = False
        for c in contours:
            if cv2.contourArea(c) > self.min_area:
                detected = True
                break
                
        self.prev_frame = gray
        return detected

class CameraStream:
    """
    Manages a single camera RTSP stream in a separate thread.
    Handles auto-reconnection and buffering.
    """
    def __init__(self, camera_config, motion_config=None):
        """
        Initialize the camera stream.
        Args:
            camera_config: Dictionary containing camera ID, URL, and name.
            motion_config: Optional dictionary for motion detection settings.
        """
        self.id = camera_config['id']
        self.url = camera_config['url']
        self.name = camera_config.get('name', self.id)
        self.reconnect_interval = 5
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_access_time = 0
        self.connected = False
        
        # Initialize motion detector if configured
        self.motion_detector = None
        if motion_config and motion_config.get('enabled', False):
            self.motion_detector = MotionDetector(
                threshold=motion_config.get('threshold', 25),
                min_area=motion_config.get('min_area', 500)
            )
        
    def start(self):
        """
        Start the capture thread.
        """
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info(f"Camera {self.name} ({self.id}) started.")

    def stop(self):
        """
        Stop the capture thread.
        """
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info(f"Camera {self.name} ({self.id}) stopped.")

    def _update(self):
        """
        Internal loop to read frames from the camera.
        Handles reconnection logic on failure.
        """
        if self.url == "test":
            # Mock mode logic
            while self.running:
                # Generate dummy frame
                frame = np.zeros((640, 640, 3), dtype=np.uint8)
                cv2.randu(frame, 0, 255)
                with self.lock:
                    self.latest_frame = frame
                    self.connected = True
                time.sleep(1/15) # 15 FPS
            return

        # Handle numeric string for webcam index
        url_to_open = self.url
        if isinstance(self.url, str) and self.url.isdigit():
            url_to_open = int(self.url)

        cap = cv2.VideoCapture(url_to_open)
        while self.running:
            if not cap.isOpened():
                self.connected = False
                logger.warning(f"Camera {self.name} disconnected. Retrying in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)
                cap = cv2.VideoCapture(url_to_open)
                continue
            
            self.connected = True
            ret, frame = cap.read()
            
            if not ret:
                self.connected = False
                logger.warning(f"Camera {self.name} failed to read frame. Reconnecting...")
                cap.release()
                time.sleep(self.reconnect_interval)
                cap = cv2.VideoCapture(url_to_open)
                continue
            
            # Store latest frame safely
            with self.lock:
                self.latest_frame = frame.copy()
            
            # Optional: Sleep to limit capture FPS if needed to save CPU, 
            # but usually we want to clear the buffer so we read as fast as possible 
            # or set CAP_PROP_BUFFERSIZE if supported.
            # time.sleep(0.01) 
            
        cap.release()

    def get_frame(self):
        """
        Retrieve the latest captured frame.
        Returns:
            numpy array or None if no frame available.
        """
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def check_motion(self, frame):
        """
        Check if motion is detected in the provided frame.
        Uses the internal MotionDetector state.
        """
        if self.motion_detector:
            return self.motion_detector.detect(frame)
        return True # If motion detection is disabled, always return True

class CameraManager:
    """
    Manages multiple CameraStream instances.
    """
    def __init__(self, config):
        """
        Initialize all configured cameras.
        Args:
            config: Full configuration dictionary.
        """
        self.cameras = {}
        motion_config = config.get('motion_detection', {})
        
        for cam_conf in config['cameras']:
            if cam_conf.get('enabled', True):
                cam = CameraStream(cam_conf, motion_config)
                self.cameras[cam.id] = cam
    
    def start_all(self):
        """
        Start all camera streams.
        """
        for cam in self.cameras.values():
            cam.start()
            
    def stop_all(self):
        """
        Stop all camera streams.
        """
        for cam in self.cameras.values():
            cam.stop()
            
    def check_motion(self, camera_id, frame):
        """
        Check if motion is detected for a specific camera.
        """
        if camera_id in self.cameras:
            return self.cameras[camera_id].check_motion(frame)
        return True

    def get_frames(self):
        """
        Get the latest frame from all active cameras.
        Returns:
            Dict {camera_id: frame}
        """
        frames = {}
        for cam_id, cam in self.cameras.items():
            frame = cam.get_frame()
            if frame is not None:
                frames[cam_id] = frame
        return frames
