import cv2
import time
import threading
import numpy as np
import logging
from queue import Queue, Empty

logger = logging.getLogger("CameraManager")

class CameraStream:
    def __init__(self, camera_config):
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

    def _update(self):
        if self.url == "test":
            while self.running:
                # Generate dummy frame
                frame = np.zeros((640, 640, 3), dtype=np.uint8)
                cv2.randu(frame, 0, 255)
                with self.lock:
                    self.latest_frame = frame
                    self.connected = True
                time.sleep(1/15) # 15 FPS
            return

        cap = cv2.VideoCapture(self.url)
        while self.running:
            if not cap.isOpened():
                self.connected = False
                logger.warning(f"Camera {self.name} disconnected. Retrying in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)
                cap = cv2.VideoCapture(self.url)
                continue
            
            self.connected = True
            ret, frame = cap.read()
            
            if not ret:
                self.connected = False
                logger.warning(f"Camera {self.name} failed to read frame. Reconnecting...")
                cap.release()
                time.sleep(self.reconnect_interval)
                cap = cv2.VideoCapture(self.url)
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
        for cam_conf in config['cameras']:
            if cam_conf.get('enabled', True):
                cam = CameraStream(cam_conf)
                self.cameras[cam.id] = cam
    
    def start_all(self):
        for cam in self.cameras.values():
            cam.start()
            
    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()
            
    def get_frames(self):
        """Returns a dict of {camera_id: frame} for all connected cameras"""
        frames = {}
        for cam_id, cam in self.cameras.items():
            frame = cam.get_frame()
            if frame is not None:
                frames[cam_id] = frame
        return frames
