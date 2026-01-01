import argparse
import logging
import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Please install it using 'pip install ultralytics'")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExportONNX")

def export_model(model_path, output_path, img_size=(640, 640)):
    """
    Export a YOLOv8 .pt model to .onnx format for Hailo compilation.
    
    Args:
        model_path (str): Path to the input .pt file.
        output_path (str): Path where the .onnx file will be saved.
        img_size (tuple): Input image size (width, height).
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False

    try:
        logger.info(f"Loading model: {model_path}")
        model = YOLO(model_path)
        
        logger.info(f"Exporting to ONNX (imgsz={img_size}, opset=11)...")
        # Hailo DFC typically supports opset 11 well
        success = model.export(format='onnx', imgsz=img_size, opset=11, dynamic=False)
        
        if success:
            logger.info(f"Export successful: {success}")
            # Ultralytics saves it in the same dir as the model usually, let's verify/move if needed
            exported_path = model_path.replace('.pt', '.onnx')
            if os.path.exists(exported_path):
                logger.info(f"ONNX model located at: {exported_path}")
            else:
                logger.warning(f"Expected ONNX path {exported_path} not found. Check ultralytics output.")
            return True
        else:
            logger.error("Export failed.")
            return False

    except Exception as e:
        logger.error(f"Export failed with exception: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX for Hailo")
    parser.add_argument("--model", type=str, default="models/ppe_best.pt", help="Path to .pt model")
    parser.add_argument("--output", type=str, default="models/ppe_best.onnx", help="Output path for .onnx model")
    
    args = parser.parse_args()
    
    export_model(args.model, args.output)
