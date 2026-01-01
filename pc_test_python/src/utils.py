import yaml
import cv2
import numpy as np
import os
import logging

def load_config(config_path):
    """
    Load YAML configuration file.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logger(name, level=logging.INFO):
    """
    Setup a standard logger with formatting.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def mask_to_polygon(mask, epsilon_factor=0.001):
    """
    Convert a binary mask to a polygon for YOLO segmentation format.
    Args:
        mask: Binary mask (numpy array).
        epsilon_factor: Approximation accuracy factor.
    Returns:
        List of normalized points [x1, y1, x2, y2, ...]
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 10: # Filter small noise
            continue
            
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Flatten and normalize
        points = approx.flatten().astype(float)
        
        # Normalize coordinates
        points[0::2] /= w
        points[1::2] /= h
        
        polygons.append(points.tolist())
        
    return polygons

def ensure_directories(base_path, subdirs):
    """
    Ensure that the specified subdirectories exist under the base path.
    Args:
        base_path: The root directory.
        subdirs: List of subdirectories (relative or absolute).
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        
    for subdir in subdirs:
        # If subdir is absolute, use it directly; otherwise join with base_path
        path = subdir if os.path.isabs(subdir) else os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)

def format_yolo_label(class_id, polygons):
    """
    Format polygons into YOLO segmentation line.
    class_id x1 y1 x2 y2 ...
    """
    lines = []
    for poly in polygons:
        line = f"{class_id} " + " ".join([f"{p:.6f}" for p in poly])
        lines.append(line)
    return lines
