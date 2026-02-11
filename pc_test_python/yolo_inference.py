import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import sys
import argparse
from pathlib import Path

def load_model(model_path):
    """Load YOLO model from ONNX or PT file"""
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def get_model_info(model):
    """Get model classes and other information"""
    try:
        if hasattr(model, 'names'):
            class_names = list(model.names.values())
            print(f"✓ Number of classes: {len(class_names)}")
            print(f"✓ Class names: {class_names}")
            return class_names
        else:
            print("⚠ Model does not have 'names' attribute")
            return []
    except Exception as e:
        print(f"✗ Error getting model info: {e}")
        return []

def run_inference(model, image_path, conf_threshold=0.5):
    """Run inference on a single image"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Error: Could not read image {image_path}")
            return None
        
        print(f"\n🖼️  Processing image: {image_path}")
        print(f"   Image shape: {image.shape}")
        
        # Run inference
        results = model(image, conf=conf_threshold)
        
        # Count detections
        num_detections = len(results[0].boxes) if hasattr(results[0], 'boxes') else 0
        print(f"   Detections found: {num_detections}")
        
        return results
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return None

def visualize_predictions(image, results, class_names, save_path=None, show_plot=True):
    """Visualize predictions with bounding boxes"""
    try:
        # Make a copy of the image for visualization
        vis_image = image.copy()
        
        # Get results
        result = results[0]
        
        detections_made = False
        
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                detections_made = True
                
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
                
                print(f"   📦 Detection {i+1}: {class_name} (confidence: {conf:.3f}) at [{x1}, {y1}, {x2}, {y2}]")
        
        if not detections_made:
            print("   ℹ️  No detections found")
        
        # Convert BGR to RGB for matplotlib
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        if show_plot:
            # Display using matplotlib
            plt.figure(figsize=(12, 8))
            plt.imshow(vis_image_rgb)
            plt.title("YOLO Predictions with Bounding Boxes")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"💾 Visualization saved to: {save_path}")
            
            plt.show()
        
        return vis_image
    except Exception as e:
        print(f"✗ Error during visualization: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="YOLO Inference with Visualization")
    parser.add_argument("--model", type=str, 
                       default=r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\models\p4_ppe.onnx",
                       help="Path to YOLO model file (.onnx or .pt)")
    parser.add_argument("--image", type=str, 
                       help="Path to input image")
    parser.add_argument("--image-dir", type=str,
                       help="Directory containing images to process")
    parser.add_argument("--output-dir", type=str, 
                       default=r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\output_predictions",
                       help="Output directory for results")
    parser.add_argument("--conf-threshold", type=float, 
                       default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--no-display", action="store_true", 
                       help="Don't display plots, only save results")
    parser.add_argument("--list-classes", action="store_true", 
                       help="Only list model classes and exit")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return 1
    
    # Get model information
    class_names = get_model_info(model)
    
    # If only listing classes, exit here
    if args.list_classes:
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Determine images to process
    images_to_process = []
    
    if args.image:
        if os.path.exists(args.image):
            images_to_process.append(args.image)
        else:
            print(f"✗ Image not found: {args.image}")
            return 1
    elif args.image_dir:
        if os.path.exists(args.image_dir):
            # Find all image files in directory
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            for file in os.listdir(args.image_dir):
                if file.lower().endswith(valid_extensions):
                    images_to_process.append(os.path.join(args.image_dir, file))
            
            if not images_to_process:
                print(f"✗ No images found in directory: {args.image_dir}")
                return 1
            
            print(f"📂 Found {len(images_to_process)} images to process")
        else:
            print(f"✗ Directory not found: {args.image_dir}")
            return 1
    else:
        # Use default sample image
        sample_image = r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\dataset_pc_test\images\train\webcam_1767265966150_e29dd4a2.jpg"
        if os.path.exists(sample_image):
            images_to_process.append(sample_image)
        else:
            # Try to find any image in the dataset
            image_dir = r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\dataset_pc_test\images\train"
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    images_to_process.append(os.path.join(image_dir, image_files[0]))
    
    if not images_to_process:
        print("✗ No images to process")
        return 1
    
    # Process images
    for i, image_path in enumerate(images_to_process, 1):
        print(f"\n{'='*60}")
        print(f"Processing image {i}/{len(images_to_process)}")
        
        # Run inference
        results = run_inference(model, image_path, args.conf_threshold)
        if results is None:
            continue
        
        # Read image for visualization
        image = cv2.imread(image_path)
        
        # Create output filename
        output_filename = f"prediction_{Path(image_path).stem}.png"
        save_path = os.path.join(args.output_dir, output_filename)
        
        # Visualize results
        visualize_predictions(image, results, class_names, save_path, show_plot=not args.no_display)
    
    print(f"\n{'='*60}")
    print(f"✅ Processing complete! Results saved to: {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())