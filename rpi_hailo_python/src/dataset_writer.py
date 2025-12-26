import os
import cv2
import time
import uuid
import random
import sqlite3
import logging
from datetime import datetime
from .utils import ensure_directories

logger = logging.getLogger("DatasetWriter")

class DatasetWriter:
    def __init__(self, config):
        self.config = config['storage']
        self.base_path = self.config['base_path']
        self.train_split = self.config.get('train_split', 0.8)
        
        self.images_dir = self.config['images_dir']
        self.labels_dir = self.config['labels_dir']
        self.db_path = os.path.join(self.base_path, self.config.get('database_path', 'datacollector.db'))
        
        self.setup_directories()
        self.setup_database()
        
    def setup_directories(self):
        # Create YOLO structure
        # dataset/images/train, dataset/images/val
        # dataset/labels/train, dataset/labels/val
        subdirs = [
            os.path.join(self.images_dir, 'train'),
            os.path.join(self.images_dir, 'val'),
            os.path.join(self.labels_dir, 'train'),
            os.path.join(self.labels_dir, 'val')
        ]
        ensure_directories(self.base_path, subdirs)
        logger.info(f"Dataset directories initialized at {self.base_path}")

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS frames (
                id TEXT PRIMARY KEY,
                camera_id TEXT,
                timestamp REAL,
                split TEXT,
                image_path TEXT,
                label_path TEXT,
                objects_count INTEGER,
                classes TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_sample(self, frame, camera_id, annotations, classes_detected):
        """
        Save a frame and its annotations.
        frame: numpy array
        camera_id: str
        annotations: list of YOLO format strings
        classes_detected: list of class names/ids found
        """
        if not annotations and not self.config.get('save_empty', False):
            return

        timestamp = time.time()
        frame_id = f"{camera_id}_{int(timestamp * 1000)}_{str(uuid.uuid4())[:8]}"
        
        # Determine split
        split = 'train' if random.random() < self.train_split else 'val'
        
        # Paths
        img_filename = f"{frame_id}.jpg"
        lbl_filename = f"{frame_id}.txt"
        
        img_rel_path = os.path.join(self.images_dir, split, img_filename)
        lbl_rel_path = os.path.join(self.labels_dir, split, lbl_filename)
        
        img_full_path = os.path.join(self.base_path, img_rel_path)
        lbl_full_path = os.path.join(self.base_path, lbl_rel_path)
        
        # Save Image
        cv2.imwrite(img_full_path, frame)
        
        # Save Label
        with open(lbl_full_path, 'w') as f:
            f.write('\n'.join(annotations))
            
        # Log to DB
        self.log_to_db(frame_id, camera_id, timestamp, split, img_rel_path, lbl_rel_path, len(annotations), str(classes_detected))
        
        logger.debug(f"Saved sample {frame_id} to {split}")

    def log_to_db(self, frame_id, camera_id, timestamp, split, img_path, lbl_path, count, classes):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO frames (id, camera_id, timestamp, split, image_path, label_path, objects_count, classes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (frame_id, camera_id, timestamp, split, img_path, lbl_path, count, classes))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log to DB: {e}")
