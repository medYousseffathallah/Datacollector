import time
import signal
import sys
import argparse
import os
import cv2
import json
from collections import deque
from datetime import datetime
from .utils import load_config, setup_logger
from .camera_manager import CameraManager
from .inference_engine import InferenceEngineJetson

logger = setup_logger("MainJetson")

class EventRecorder:
    def __init__(self, config):
        detection = config.get('detection', {})
        self.pre_seconds = float(detection.get('pre_seconds', 15.0))
        self.post_seconds = float(detection.get('post_seconds', 15.0))
        self.clip_fps = float(detection.get('clip_fps', 15.0))
        self.output_dir = detection.get('output_dir', 'events')
        self.trigger_classes = detection.get('trigger_classes', [])
        self.require_person = bool(detection.get('require_person', False))
        self.person_class_name = detection.get('person_class_name', 'Person')
        self.per_person = bool(detection.get('per_person', True))
        self.cooldown_seconds = float(detection.get('cooldown_seconds', 0.0))
        self.crop_padding = int(detection.get('crop_padding', 10))
        self.timestamp_format = detection.get('timestamp_format', '%Y%m%d_%H%M%S_%f')
        self.fourcc = detection.get('fourcc', 'mp4v')
        self.time_windows = detection.get('time_windows', [])
        self.buffers = {}
        self.active_clips = {}
        self.last_event_times = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def is_time_allowed(self, timestamp):
        if not self.time_windows:
            return True
        dt = datetime.fromtimestamp(timestamp)
        minutes = dt.hour * 60 + dt.minute
        weekday = dt.weekday()
        for window in self.time_windows:
            start_s = window.get('start')
            end_s = window.get('end')
            if not start_s or not end_s:
                continue
            days = window.get('days')
            if days is not None:
                normalized_days = []
                for d in days:
                    if isinstance(d, int):
                        normalized_days.append(d)
                    elif isinstance(d, str):
                        name = d.strip().lower()
                        mapping = {
                            'mon': 0,
                            'monday': 0,
                            'tue': 1,
                            'tuesday': 1,
                            'wed': 2,
                            'wednesday': 2,
                            'thu': 3,
                            'thursday': 3,
                            'fri': 4,
                            'friday': 4,
                            'sat': 5,
                            'saturday': 5,
                            'sun': 6,
                            'sunday': 6,
                        }
                        if name in mapping:
                            normalized_days.append(mapping[name])
                if normalized_days and weekday not in normalized_days:
                    continue

            try:
                start_h, start_m = start_s.split(':', 1)
                end_h, end_m = end_s.split(':', 1)
                start_minutes = int(start_h) * 60 + int(start_m)
                end_minutes = int(end_h) * 60 + int(end_m)
            except Exception:
                continue

            if start_minutes <= end_minutes:
                if start_minutes <= minutes <= end_minutes:
                    return True
            else:
                if minutes >= start_minutes or minutes <= end_minutes:
                    return True
        return False

    def update_buffer(self, camera_id, frame, timestamp):
        buffer = self.buffers.setdefault(camera_id, deque())
        buffer.append((timestamp, frame.copy()))
        cutoff = timestamp - self.pre_seconds
        while buffer and buffer[0][0] < cutoff:
            buffer.popleft()

    def write_active(self, camera_id, frame, timestamp):
        if camera_id not in self.active_clips:
            return
        remaining = []
        for clip in self.active_clips[camera_id]:
            if timestamp <= clip['end_time']:
                crop = self._crop_frame(frame, clip['bbox'])
                if crop is not None:
                    clip['writer'].write(crop)
                remaining.append(clip)
            else:
                clip['writer'].release()
        if remaining:
            self.active_clips[camera_id] = remaining
        else:
            del self.active_clips[camera_id]

    def process_detections(self, camera_id, frame, timestamp, detections):
        if not self.is_time_allowed(timestamp):
            return
        if not detections:
            return
        persons = [d for d in detections if d['class_name'] == self.person_class_name]
        triggers = detections if not self.trigger_classes else [d for d in detections if d['class_name'] in self.trigger_classes]
        if not triggers:
            return
        if self.require_person and not persons:
            return
        if self.per_person and persons:
            for idx, person in enumerate(persons):
                self._start_clip(camera_id, timestamp, person['bbox'], triggers[0]['class_name'], idx)
        else:
            target = persons[0] if persons else triggers[0]
            self._start_clip(camera_id, timestamp, target['bbox'], target['class_name'], 0)

    def _start_clip(self, camera_id, timestamp, bbox, class_name, person_index):
        if bbox is None:
            return
        event_key = (camera_id, class_name, person_index)
        last_time = self.last_event_times.get(event_key)
        if last_time is not None and (timestamp - last_time) < self.cooldown_seconds:
            return
        self.last_event_times[event_key] = timestamp
        crop = self._crop_frame(self._get_latest_frame(camera_id), bbox)
        if crop is None:
            return
        height, width = crop.shape[:2]
        if height < 2 or width < 2:
            return
        cam_dir = os.path.join(self.output_dir, camera_id)
        os.makedirs(cam_dir, exist_ok=True)
        timestamp_str = datetime.fromtimestamp(timestamp).strftime(self.timestamp_format)
        safe_class = class_name.replace(' ', '_')
        filename = f"{camera_id}_{safe_class}_p{person_index}_{timestamp_str}.mp4"
        output_path = os.path.join(cam_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        writer = cv2.VideoWriter(output_path, fourcc, self.clip_fps, (width, height))
        pre_frames = list(self.buffers.get(camera_id, []))
        for t, f in pre_frames:
            if t >= timestamp - self.pre_seconds:
                crop_frame = self._crop_frame(f, bbox)
                if crop_frame is not None:
                    writer.write(crop_frame)
        clip = {
            'writer': writer,
            'end_time': timestamp + self.post_seconds,
            'bbox': bbox,
            'output_path': output_path
        }
        self.active_clips.setdefault(camera_id, []).append(clip)
        self._write_event_json(output_path, camera_id, timestamp, class_name, person_index, bbox)

    def _get_latest_frame(self, camera_id):
        buffer = self.buffers.get(camera_id)
        if not buffer:
            return None
        return buffer[-1][1]

    def _crop_frame(self, frame, bbox):
        if frame is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = self.crop_padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _write_event_json(self, output_path, camera_id, timestamp, class_name, person_index, bbox):
        json_path = os.path.splitext(output_path)[0] + '.json'
        payload = {
            'camera_id': camera_id,
            'timestamp_epoch': timestamp,
            'timestamp_local': datetime.fromtimestamp(timestamp).isoformat(),
            'class_name': class_name,
            'person_index': person_index,
            'bbox': list(bbox) if bbox else None,
            'clip_path': output_path,
            'pre_seconds': self.pre_seconds,
            'post_seconds': self.post_seconds,
            'clip_fps': self.clip_fps,
            'trigger_classes': self.trigger_classes,
            'require_person': self.require_person,
            'per_person': self.per_person,
            'time_windows': self.time_windows
        }
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)

class DataCollectorJetson:
    """
    Main controller for the Jetson version of the Data Collector.
    Uses InferenceEngineJetson for CUDA-accelerated inference.
    """
    def __init__(self, config_path):
        """
        Initialize the DataCollector with configuration.
        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config = load_config(config_path)
        self.running = True
        
        # Init Components
        self.camera_manager = CameraManager(self.config)
        # Use Jetson Engine
        self.inference_engine = InferenceEngineJetson(self.config)
        self.event_recorder = EventRecorder(self.config)
        self.class_names = self.config['inference'].get('class_names', [])
        # If class_names empty, try to get from model names
        if not self.class_names and self.inference_engine.model:
            self.class_names = self.inference_engine.model.names
        self.min_confidence = self.config.get('detection', {}).get('min_confidence', 0.6)
        self.process_interval = float(self.config.get('detection', {}).get('process_interval_seconds', 0.0))
        self.last_process_times = {}

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        """
        Signal handler for graceful shutdown.
        """
        logger.info("Shutdown signal received...")
        self.running = False

    def run(self):
        """
        Main execution loop.
        """
        logger.info("Starting Data Collector System (Jetson Edition)...")
        
        self.camera_manager.start_all()
        self.inference_engine.start()
        
        logger.info("System running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                frames = self.camera_manager.get_frames()
                current_time = time.time()

                for cam_id, frame in frames.items():
                    last_time = self.last_process_times.get(cam_id, 0)
                    if current_time - last_time < self.process_interval:
                        self.event_recorder.update_buffer(cam_id, frame, current_time)
                        self.event_recorder.write_active(cam_id, frame, current_time)
                        continue

                    if not self.event_recorder.is_time_allowed(current_time):
                        self.event_recorder.update_buffer(cam_id, frame, current_time)
                        self.event_recorder.write_active(cam_id, frame, current_time)
                        self.last_process_times[cam_id] = current_time
                        continue
                    
                    # Check for motion if configured
                    if not self.camera_manager.check_motion(cam_id, frame):
                        # No motion detected, skip inference to save resources
                        continue
                        
                    # Inference
                    results = self.inference_engine.infer(frame)
                    if not results:
                        self.event_recorder.update_buffer(cam_id, frame, current_time)
                        self.event_recorder.write_active(cam_id, frame, current_time)
                        continue
                        
                    masks, class_ids, scores = results
                    detections = []
                    for mask, cls_id, score in zip(masks, class_ids, scores):
                        if score < self.min_confidence:
                            continue
                        if isinstance(self.class_names, dict):
                            class_name = self.class_names.get(cls_id, str(cls_id))
                        elif cls_id < len(self.class_names):
                            class_name = self.class_names[cls_id]
                        else:
                            class_name = str(cls_id)
                        bbox = self._mask_to_bbox(mask)
                        detections.append({
                            'class_id': cls_id,
                            'class_name': class_name,
                            'score': score,
                            'bbox': bbox
                        })
                    self.event_recorder.update_buffer(cam_id, frame, current_time)
                    self.event_recorder.process_detections(cam_id, frame, current_time, detections)
                    self.event_recorder.write_active(cam_id, frame, current_time)
                    self.last_process_times[cam_id] = current_time
                
                time.sleep(0.01) # Faster poll on Jetson
                
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Stop services and release resources.
        """
        logger.info("Cleaning up resources...")
        self.camera_manager.stop_all()
        self.inference_engine.stop()
        logger.info("Shutdown complete.")

    def _mask_to_bbox(self, mask):
        if mask is None:
            return None
        ys, xs = (mask > 0).nonzero()
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return int(x1), int(y1), int(x2), int(y2)

def main():
    """
    Entry point of the application.
    """
    parser = argparse.ArgumentParser(description="Jetson AI Data Collector")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    collector = DataCollectorJetson(args.config)
    collector.run()

if __name__ == "__main__":
    main()
