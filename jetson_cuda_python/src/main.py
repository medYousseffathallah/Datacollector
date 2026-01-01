import time
import signal
import sys
import argparse
import os
import cv2
from .utils import load_config, setup_logger, mask_to_polygon, format_yolo_label
from .camera_manager import CameraManager
from .dataset_writer import DatasetWriter
# Specific Import for Jetson
from .inference_engine import InferenceEngineJetson

logger = setup_logger("MainJetson")

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
        self.dataset_writer = DatasetWriter(self.config)
        
        self.class_names = self.config['inference'].get('class_names', [])
        # If class_names empty, try to get from model names
        if not self.class_names and self.inference_engine.model:
            self.class_names = self.inference_engine.model.names
            
        self.target_classes = set(self.config['collection'].get('target_classes', []))
        self.min_confidence = self.config['collection'].get('min_confidence', 0.6)
        
        self.capture_interval = self.config['collection'].get('interval_seconds', 5.0)
        self.last_capture_times = {} 

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
                    last_time = self.last_capture_times.get(cam_id, 0)
                    if current_time - last_time < self.capture_interval:
                        continue
                    
                    # Check for motion if configured
                    if not self.camera_manager.check_motion(cam_id, frame):
                        # No motion detected, skip inference to save resources
                        continue
                        
                    # Inference
                    results = self.inference_engine.infer(frame)
                    if not results:
                        continue
                        
                    masks, class_ids, scores = results
                    
                    yolo_annotations = []
                    classes_detected = []
                    
                    for mask, cls_id, score in zip(masks, class_ids, scores):
                        if score < self.min_confidence:
                            continue
                            
                        # Handle class names map
                        if isinstance(self.class_names, dict):
                             class_name = self.class_names.get(cls_id, str(cls_id))
                        elif cls_id < len(self.class_names):
                            class_name = self.class_names[cls_id]
                        else:
                            class_name = str(cls_id)
                        
                        if self.target_classes and class_name not in self.target_classes:
                            continue
                            
                        polygons = mask_to_polygon(mask)
                        if not polygons:
                            continue
                            
                        lines = format_yolo_label(cls_id, polygons)
                        yolo_annotations.extend(lines)
                        classes_detected.append(class_name)
                    
                    if yolo_annotations:
                        self.dataset_writer.save_sample(frame, cam_id, yolo_annotations, classes_detected)
                        self.last_capture_times[cam_id] = current_time
                        logger.info(f"Captured sample from {cam_id}: {len(yolo_annotations)} objects")
                
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
