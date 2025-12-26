import time
import signal
import sys
import argparse
import os
import cv2
from .utils import load_config, setup_logger, mask_to_polygon, format_yolo_label
from .camera_manager import CameraManager
from .inference_engine import InferenceEngine
from .dataset_writer import DatasetWriter

logger = setup_logger("Main")

class DataCollector:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.running = True
        
        # Init Components
        self.camera_manager = CameraManager(self.config)
        self.inference_engine = InferenceEngine(self.config)
        self.dataset_writer = DatasetWriter(self.config)
        
        self.class_names = self.config['inference'].get('class_names', [])
        self.target_classes = set(self.config['collection'].get('target_classes', []))
        self.min_confidence = self.config['collection'].get('min_confidence', 0.6)
        
        self.capture_interval = self.config['collection'].get('interval_seconds', 5.0)
        self.last_capture_times = {} # {camera_id: timestamp}

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        logger.info("Shutdown signal received...")
        self.running = False

    def run(self):
        logger.info("Starting Data Collector System...")
        
        # Start Cameras
        self.camera_manager.start_all()
        
        # Start Inference Engine
        self.inference_engine.start()
        
        logger.info("System running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                frames = self.camera_manager.get_frames()
                current_time = time.time()
                
                for cam_id, frame in frames.items():
                    # Check interval
                    last_time = self.last_capture_times.get(cam_id, 0)
                    if current_time - last_time < self.capture_interval:
                        continue
                        
                    # Inference
                    results = self.inference_engine.infer(frame)
                    if not results:
                        continue
                        
                    masks, class_ids, scores = results
                    
                    # Filter and Format
                    yolo_annotations = []
                    classes_detected = []
                    
                    for mask, cls_id, score in zip(masks, class_ids, scores):
                        if score < self.min_confidence:
                            continue
                            
                        class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                        
                        if self.target_classes and class_name not in self.target_classes:
                            continue
                            
                        # Convert Mask to Polygon
                        polygons = mask_to_polygon(mask)
                        if not polygons:
                            continue
                            
                        # Format Label
                        lines = format_yolo_label(cls_id, polygons)
                        yolo_annotations.extend(lines)
                        classes_detected.append(class_name)
                    
                    # Save if we have annotations (or if configured to save empty)
                    if yolo_annotations:
                        self.dataset_writer.save_sample(frame, cam_id, yolo_annotations, classes_detected)
                        self.last_capture_times[cam_id] = current_time
                        logger.info(f"Captured sample from {cam_id}: {len(yolo_annotations)} objects")
                
                # Sleep to prevent tight loop burning CPU
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up resources...")
        self.camera_manager.stop_all()
        self.inference_engine.stop()
        logger.info("Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="Edge AI Data Collector for YOLO Fine-tuning")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    collector = DataCollector(args.config)
    collector.run()

if __name__ == "__main__":
    main()
