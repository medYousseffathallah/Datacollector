import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import sys
from pathlib import Path

def load_model(model_path):
    """Load YOLO model from ONNX or PT file"""
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
        print(f"Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_model_info(model):
    """Get model classes and other information"""
    try:
        if hasattr(model, 'names'):
            class_names = list(model.names.values())
            print(f"Number of classes: {len(class_names)}")
            print(f"Class names: {class_names}")
            return class_names
        else:
            print("Model does not have 'names' attribute")
            return []
    except Exception as e:
        print(f"Error getting model info: {e}")
        return []

def run_inference(model, image_path, conf_threshold=0.5):
    """Run inference on a single image"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Run inference
        results = model(image, conf=conf_threshold)
        
        print(f"Inference completed for: {image_path}")
        print(f"Number of detections: {len(results[0].boxes) if hasattr(results[0], 'boxes') else 0}")
        
        return results
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def visualize_predictions(image, results, class_names, save_path=None):
    """Visualize predictions with bounding boxes"""
    try:
        # Make a copy of the image for visualization
        vis_image = image.copy()
        
        # Get results
        result = results[0]
        
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get confidence and class
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                print(f"Detection {i+1}: {class_name} (confidence: {conf:.3f}) at [{x1}, {y1}, {x2}, {y2}]")
        
        # Convert BGR to RGB for matplotlib
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        # Display using matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image_rgb)
        plt.title("YOLO Predictions with Bounding Boxes")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return vis_image
    except Exception as e:
        print(f"Error during visualization: {e}")
        return None

def process_single_image(model, image_path, class_names, output_dir=None, conf_threshold=0.5):
    """Process a single image and show results"""
    print(f"\nProcessing image: {image_path}")
    
    # Run inference
    results = run_inference(model, image_path, conf_threshold)
    if results is None:
        return
    
    # Read image for visualization
    image = cv2.imread(image_path)
    
    # Visualize results
    if output_dir:
        save_path = os.path.join(output_dir, f"prediction_{Path(image_path).stem}.png")
        visualize_predictions(image, results, class_names, save_path)
    else:
        visualize_predictions(image, results, class_names)
    
    # Print detailed results
    print("\nDetailed Detection Results:")
    if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            print(f"  {i+1}. {class_name} (confidence: {conf:.3f}, bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}])")
    else:
        print("  No detections found")

def main():
    # Model path
    model_path = r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\models\p4_ppe.onnx"
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Get model information
    class_names = get_model_info(model)
    
    # Use a sample image from the dataset
    sample_image = r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\dataset_pc_test\images\train\webcam_1767265966150_e29dd4a2.jpg"
    
    if not os.path.exists(sample_image):
        print(f"Sample image not found: {sample_image}")
        # Try to find any image in the dataset
        image_dir = r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\dataset_pc_test\images\train"
        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                sample_image = os.path.join(image_dir, image_files[0])
                print(f"Using alternative image: {sample_image}")
            else:
                print("No images found in dataset directory")
                return
        else:
            print("Dataset directory not found")
            return
    
    # Process the image
    output_dir = r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\output_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    process_single_image(model, sample_image, class_names, output_dir, conf_threshold=0.5)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()