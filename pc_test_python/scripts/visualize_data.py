import streamlit as st
import pandas as pd
import sqlite3
import os
import yaml
import cv2
import numpy as np
from PIL import Image

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_db_connection(db_path):
    return sqlite3.connect(db_path)

def draw_yolo_labels(image, label_path, class_names):
    """
    Draw YOLO segmentation polygons or boxes on the image.
    """
    if not os.path.exists(label_path):
        return image

    img_h, img_w = image.shape[:2]
    
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Create a copy to draw on
    img_draw = image.copy()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        
        color = (0, 255, 0) # Green default
        # Simple color cycle based on class_id
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        color = colors[class_id % len(colors)]

        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

        if len(coords) == 4:
            # Bounding Box (cx, cy, w, h)
            cx, cy, w, h = coords
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_draw, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        elif len(coords) > 4:
            # Polygon
            # Reshape to (-1, 2)
            points = np.array(coords).reshape(-1, 2)
            # Denormalize
            points[:, 0] *= img_w
            points[:, 1] *= img_h
            points = points.astype(np.int32)
            
            cv2.polylines(img_draw, [points], isClosed=True, color=color, thickness=2)
            # Put text at the first point
            cv2.putText(img_draw, class_name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_draw

def main():
    st.set_page_config(page_title="Data Collector Viewer", layout="wide")
    st.title("Edge AI Data Collector - Dataset Viewer")

    # Sidebar Config
    config_path = st.sidebar.text_input("Config Path", "config/config.yaml")
    
    if not os.path.exists(config_path):
        st.error(f"Config file not found at {config_path}")
        return

    config = load_config(config_path)
    
    # Paths
    base_path = config['storage']['base_path']
    db_rel_path = config['storage'].get('database_path', 'datacollector.db')
    db_path = os.path.join(base_path, db_rel_path)
    images_dir = os.path.join(base_path, config['storage']['images_dir'])
    labels_dir = os.path.join(base_path, config['storage']['labels_dir'])
    class_names = config['inference'].get('class_names', [])

    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}. Run the data collector first.")
        return

    # Mode Selection
    mode = st.sidebar.radio("Mode", ["Dashboard", "Gallery", "Database View"])

    conn = get_db_connection(db_path)

    if mode == "Dashboard":
        st.header("Collection Statistics")
        
        # Load data
        df = pd.read_sql_query("SELECT * FROM frames", conn)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Frames", len(df))
        col2.metric("Training Samples", len(df[df['split'] == 'train']))
        col3.metric("Validation Samples", len(df[df['split'] == 'val']))
        
        st.subheader("Objects per Class")
        # This is a bit complex because 'classes' is a string representation of a list
        # simplified check
        st.dataframe(df)

    elif mode == "Gallery":
        st.header("Image Gallery")
        
        split = st.sidebar.selectbox("Split", ["train", "val"])
        
        # Get images from DB or filesystem
        # Filesystem is more reliable for what actually exists
        split_img_dir = os.path.join(images_dir, split)
        split_lbl_dir = os.path.join(labels_dir, split)
        
        if not os.path.exists(split_img_dir):
            st.warning(f"No directory found: {split_img_dir}")
            return

        image_files = sorted([f for f in os.listdir(split_img_dir) if f.endswith(('.jpg', '.png'))])
        
        if not image_files:
            st.info("No images found in this split.")
            return

        # Pagination
        page_size = 8
        page_number = st.sidebar.number_input("Page", min_value=1, value=1)
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        
        current_batch = image_files[start_idx:end_idx]
        
        cols = st.columns(4)
        for i, img_file in enumerate(current_batch):
            img_path = os.path.join(split_img_dir, img_file)
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            lbl_path = os.path.join(split_lbl_dir, lbl_file)
            
            # Load and draw
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_annotated = draw_yolo_labels(img, lbl_path, class_names)
            
            with cols[i % 4]:
                st.image(img_annotated, caption=img_file, use_container_width=True)
                # Show label content expander
                if os.path.exists(lbl_path):
                    with st.expander("Label Data"):
                        with open(lbl_path, 'r') as f:
                            st.text(f.read())

    elif mode == "Database View":
        st.header("Raw Database Data")
        query = st.text_area("SQL Query", "SELECT * FROM frames ORDER BY timestamp DESC LIMIT 100")
        try:
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Query Error: {e}")

    conn.close()

if __name__ == "__main__":
    main()
