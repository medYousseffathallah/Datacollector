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

# Initialize the logger for the main module
logger = setup_logger("Main")

class DataCollector:
    """
    Main controller class for the Data Collector system.
    Orchestrates camera capture, inference, and data saving.
    """
    def __init__(self, config_path):
        """
        Initialize the DataCollector with configuration.
        Args:
            config_path: Path to the YAML configuration file.
        """
        # Load configuration from file
        self.config = load_config(config_path)
        self.running = True
        
        # Init Components
        # Initialize Camera Manager to handle RTSP streams
        self.camera_manager = CameraManager(self.config)
        # Initialize Inference Engine for HailoRT
        self.inference_engine = InferenceEngine(self.config)
        # Initialize Dataset Writer for saving images/labels
        self.dataset_writer = DatasetWriter(self.config)
        
        # Load class names mapping
        self.class_names = self.config['inference'].get('class_names', [])
        # Load target classes to filter (if any)
        self.target_classes = set(self.config['collection'].get('target_classes', []))
        # Load minimum confidence threshold
        self.min_confidence = self.config['collection'].get('min_confidence', 0.6)
        
        # Capture interval in seconds
        self.capture_interval = self.config['collection'].get('interval_seconds', 5.0)
        self.last_capture_times = {} # Dictionary to track last capture time per camera

        # Register signal handlers for graceful shutdown
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
        logger.info("Starting Data Collector System...")
        
        # Start all configured cameras
        self.camera_manager.start_all()
        
        # Start the inference engine (activates Hailo network)
        self.inference_engine.start()
        
        logger.info("System running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                # Retrieve the latest frames from all connected cameras
                frames = self.camera_manager.get_frames()
                current_time = time.time()
                
                for cam_id, frame in frames.items():
                    # Check if enough time has passed since last capture for this camera
                    last_time = self.last_capture_times.get(cam_id, 0)
                    if current_time - last_time < self.capture_interval:
                        continue
                        
                    # Run inference on the current frame
                    results = self.inference_engine.infer(frame)
                    if not results:
                        continue
                        
                    # Unpack results: binary masks, class IDs, and confidence scores
                    masks, class_ids, scores = results
                    
                    # Lists to hold formatted annotations and detected class names
                    yolo_annotations = []
                    classes_detected = []
                    
                    for mask, cls_id, score in zip(masks, class_ids, scores):
                        # Filter detections below confidence threshold
                        if score < self.min_confidence:
                            continue
                            
                        # Get class name from ID
                        class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                        
                        # Filter by target class if whitelist is configured
                        if self.target_classes and class_name not in self.target_classes:
                            continue
                            
                        # Convert binary mask to normalized polygon coordinates
                        polygons = mask_to_polygon(mask)
                        if not polygons:
                            continue
                            
                        # Format polygon into YOLO segmentation label format
                        lines = format_yolo_label(cls_id, polygons)
                        yolo_annotations.extend(lines)
                        classes_detected.append(class_name)
                    
                    # Save the sample if valid annotations were found (or if configured to save empty frames)
                    if yolo_annotations:
                        self.dataset_writer.save_sample(frame, cam_id, yolo_annotations, classes_detected)
                        self.last_capture_times[cam_id] = current_time
                        logger.info(f"Captured sample from {cam_id}: {len(yolo_annotations)} objects")
                
                # Sleep briefly to prevent high CPU usage in the main loop
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            # Ensure resources are cleaned up on exit
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
    parser = argparse.ArgumentParser(description="Edge AI Data Collector for YOLO Fine-tuning")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    # Create and run the collector
    collector = DataCollector(args.config)
    collector.run()

if __name__ == "__main__":
    main()
