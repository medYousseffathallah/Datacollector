from ultralytics import YOLO
import os

def convert_model():
    # Path to the input model
    model_path = 'jetson_cuda_python/models/ppe_best.pt'
    output_dir = 'jetson_cuda_python/models'
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the YOLO model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Export to ONNX format
    print("Converting model to ONNX format...")
    export_result = model.export(format='onnx', 
                               imgsz=640,  # default YOLOv8 input size
                               opset=12,   # ONNX opset version
                               simplify=True)  # simplify model
    
    print(f"Conversion successful! Output file: {export_result}")
    
    # Verify the output file
    output_path = os.path.join(output_dir, 'ppe_best.onnx')
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Output file size: {file_size:.2f} MB")
    else:
        print("Warning: ONNX file not found in expected location")

if __name__ == "__main__":
    try:
        convert_model()
    except Exception as e:
        print(f"Error converting model: {e}")
        raise