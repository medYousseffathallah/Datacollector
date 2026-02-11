import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import sys
from pathlib import Path

def quick_inference_demo():
    """Quick demo of YOLO inference with visualization"""
    
    # Model path
    model_path = r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\models\ppe_best.onnx"
    
    print("🚀 YOLO PPE Detection Demo")
    print("="*50)
    
    # Load model
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get class names
    if hasattr(model, 'names'):
        class_names = list(model.names.values())
        print(f"📋 Available classes ({len(class_names)}): {class_names}")
    else:
        class_names = []
    
    # Find sample images
    image_dir = r"C:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\dataset_pc_test\images\test12"
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    # Get first few images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("❌ No images found in directory")
        return
    
    print(f"📁 Found {len(image_files)} images in dataset")
    
    # Process first 3 images
    output_dir = r"c:\Users\youss\OneDrive\Desktop\project\datacollector\pc_test_python\quick_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image_file in enumerate(image_files[:3]):
        image_path = os.path.join(image_dir, image_file)
        print(f"\n🖼️  Processing image {i+1}/3: {image_file}")
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  Could not read image: {image_path}")
                continue
            
            # Run inference
            results = model(image, conf=0.1)
            
            # Get detections
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                print(f"   Found {len(boxes)} detections:")
                
                # Draw bounding boxes
                vis_image = image.copy()
                
                for j, box in enumerate(boxes):
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
                    cv2.putText(vis_image, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    print(f"     {j+1}. {class_name} ({conf:.3f})")
                
                # Save result
                output_path = os.path.join(output_dir, f"demo_result_{i+1}.png")
                cv2.imwrite(output_path, vis_image)
                print(f"   💾 Saved result to: {output_path}")
                
            else:
                print("   No detections found")
                
        except Exception as e:
            print(f"⚠️  Error processing image: {e}")
    
    print(f"\n✅ Demo complete! Results saved to: {output_dir}")
    print("\nTo run custom inference, use:")
    print("python yolo_inference.py --image path/to/your/image.jpg")
    print("python yolo_inference.py --image-dir path/to/image/directory")

if __name__ == "__main__":
    quick_inference_demo()