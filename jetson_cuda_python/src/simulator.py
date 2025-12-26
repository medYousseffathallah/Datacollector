import os
import sys
import time
import logging
import shutil
import threading
import yaml
from .main import DataCollectorJetson
from .utils import load_config

# Configure logging for simulator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SIMULATOR - %(levelname)s - %(message)s')
logger = logging.getLogger("Simulator")

def check_environment():
    """
    Check if required dependencies and folders exist for Jetson environment.
    """
    logger.info("Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required.")
        return False
        
    # Check dependencies
    try:
        import cv2
        import numpy
        import yaml
        logger.info("Core dependencies (cv2, numpy, yaml) found.")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

    # Check directories
    required_dirs = ['config', 'src', 'dataset']
    for d in required_dirs:
        if not os.path.exists(d):
            logger.error(f"Missing directory: {d}")
            return False
            
    logger.info("Environment check passed.")
    return True

def run_simulation(duration=15):
    """
    Run the DataCollectorJetson in a simulated environment.
    """
    logger.info(f"Starting simulation for {duration} seconds...")
    
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        logger.error("Config file not found.")
        return False

    # Load and patch config for simulation
    config = load_config(config_path)
    
    # Force Mock Cameras
    config['cameras'] = [
        {
            'id': 'sim_cam_01',
            'url': 'test', # Triggers internal test generator
            'name': 'SimulatorCam1',
            'enabled': True
        },
        {
            'id': 'sim_cam_02',
            'url': 'test',
            'name': 'SimulatorCam2',
            'enabled': True
        }
    ]
    
    # Force fast capture interval for testing
    config['collection']['interval_seconds'] = 1.0
    
    # Write temp config
    temp_config_path = 'config/config_sim.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
        
    # Initialize Collector with patched config
    collector = DataCollectorJetson(temp_config_path)
    
    # Run in separate thread
    collector_thread = threading.Thread(target=collector.run, daemon=True)
    collector_thread.start()
    
    # Wait for duration
    time.sleep(duration)
    
    # Stop collector
    collector.shutdown(None, None)
    collector_thread.join(timeout=5)
    
    # Cleanup temp config
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        
    logger.info("Simulation run finished.")
    return True

def verify_outputs():
    """
    Check if the simulation generated valid output files.
    """
    logger.info("Verifying outputs...")
    
    base_path = 'dataset'
    images_train = os.path.join(base_path, 'images', 'train')
    labels_train = os.path.join(base_path, 'labels', 'train')
    db_path = os.path.join(base_path, 'datacollector.db')
    
    # Check if DB exists
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return False
        
    # Check if files generated
    has_images = False
    for root, dirs, files in os.walk(os.path.join(base_path, 'images')):
        if any(f.endswith(('.jpg', '.png')) for f in files):
            has_images = True
            break
            
    has_labels = False
    for root, dirs, files in os.walk(os.path.join(base_path, 'labels')):
        if any(f.endswith('.txt') for f in files):
            has_labels = True
            break
            
    if has_images and has_labels:
        logger.info("Output verification passed: Images and Labels found.")
        return True
    else:
        logger.error(f"Output verification failed. Images: {has_images}, Labels: {has_labels}")
        return False

def main():
    logger.info("=== Data Collector Simulator (Jetson) ===")
    
    if not check_environment():
        logger.error("Environment check failed. Aborting.")
        sys.exit(1)
        
    try:
        if run_simulation():
            if verify_outputs():
                logger.info("=== SIMULATION SUCCESSFUL ===")
                logger.info("The system is correctly configured and functional.")
            else:
                logger.error("=== SIMULATION FAILED: No output generated ===")
                sys.exit(1)
        else:
            logger.error("=== SIMULATION FAILED: Runtime error ===")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Simulation crashed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
