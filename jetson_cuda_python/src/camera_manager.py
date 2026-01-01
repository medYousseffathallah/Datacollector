import cv2
import time
import threading
import numpy as np
import logging
from queue import Queue, Empty

logger = logging.getLogger("CameraManager")

class MotionDetector:
    def __init__(self, config):
        self.enabled = config.get('enabled', False)
        self.threshold = config.get('threshold', 25)
        self.min_area = config.get('min_area', 500)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    def detect(self, frame):
        if not self.enabled:
            return True

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                return True
        
        return False

class CameraStream:
    def __init__(self, camera_config, motion_config=None):
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
        
        # Motion detection
        self.motion_detector = MotionDetector(motion_config if motion_config else {})
        self.motion_detected = False
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info(f"Camera {self.name} ({self.id}) started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info(f"Camera {self.name} ({self.id}) stopped.")

    def check_motion(self, frame):
        """
        Check if motion is detected in the frame.
        """
        if not self.motion_detector.enabled:
            return True
        return self.motion_detector.detect(frame)

    def _update(self):
        # Handle numeric string for webcam index
        url_to_open = self.url
        if isinstance(self.url, str) and self.url.isdigit():
            url_to_open = int(self.url)

        if url_to_open == "test":
            while self.running:
                # Generate dummy frame
                frame = np.zeros((640, 640, 3), dtype=np.uint8)
                cv2.randu(frame, 0, 255)
                with self.lock:
                    self.latest_frame = frame
                    self.connected = True
                time.sleep(1/15) # 15 FPS
            return

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
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

class CameraManager:
    def __init__(self, config):
        self.cameras = {}
        motion_config = config.get('motion_detection', {})
        for cam_conf in config['cameras']:
            if cam_conf.get('enabled', True):
                cam = CameraStream(cam_conf, motion_config)
                self.cameras[cam.id] = cam
    
    def start_all(self):
        for cam in self.cameras.values():
            cam.start()
            
    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()
            
    def get_frames(self):
        frames = {}
        for cam_id, cam in self.cameras.items():
            frame = cam.get_frame()
            if frame is not None:
                frames[cam_id] = frame
        return frames

    def check_motion(self, camera_id, frame):
        """
        Check if motion is detected for a specific camera.
        """
        if camera_id in self.cameras:
            return self.cameras[camera_id].check_motion(frame)
        return True
            
    def get_frames(self):
        """Returns a dict of {camera_id: frame} for all connected cameras"""
        frames = {}
        for cam_id, cam in self.cameras.items():
            frame = cam.get_frame()
            if frame is not None:
                frames[cam_id] = frame
        return frames
