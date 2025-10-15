# In src/coco_dataset.py
import os
import json
import cv2
import numpy as np
from collections import defaultdict

class CocoDataset:
    """
    Custom MindSpore dataset loader for the COCO 2017 object detection format,
    now with filtering for specific classes.
    """
    def __init__(self, annotations_file, images_dir):
        print("Initializing COCO dataset loader...")
        self.images_dir = images_dir
        
        # 1. Define your 8 target classes from your proposal
        target_classes = {'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'}

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # 2. Get the original COCO category IDs for your target classes
        target_cat_ids = set()
        self.cat_id_to_index = {}
        new_class_index = 0
        for cat in coco_data['categories']:
            if cat['name'] in target_classes:
                target_cat_ids.add(cat['id'])
                # Create a new mapping from the original COCO ID to your new 0-7 index
                self.cat_id_to_index[cat['id']] = new_class_index
                new_class_index += 1
        
        print(f"Found {len(target_cat_ids)} target category IDs.")

        # 3. Filter annotations to only include your target classes
        #    Also, create a set of image IDs that contain at least one target object.
        self.annotations = defaultdict(list)
        images_with_targets = set()
        
        print("Filtering annotations...")
        for ann in coco_data['annotations']:
            if ann['category_id'] in target_cat_ids:
                self.annotations[ann['image_id']].append(ann)
                images_with_targets.add(ann['image_id'])

        # 4. Filter images to only include those that have at least one target object
        self.images = []
        print("Filtering images...")
        for img in coco_data['images']:
            if img['id'] in images_with_targets:
                self.images.append(img)
        
        print(f"Dataset filtered. Kept {len(self.images)} images containing target classes.")

    def __getitem__(self, index):
        image_info = self.images[index]
        image_id = image_info['id']
        file_name = image_info['file_name']
        
        image_path = os.path.join(self.images_dir, file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations_for_image = self.annotations[image_id]
        
        boxes = []
        labels = []
        
        for ann in annotations_for_image:
            x, y, w, h = ann['bbox']
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            boxes.append([xmin, ymin, xmax, ymax])
            
            # Use the new mapping to get a label index from 0-7
            label_index = self.cat_id_to_index[ann['category_id']]
            labels.append(label_index)

        return (np.array(image, dtype=np.uint8),
                np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int32))

    def __len__(self):
        return len(self.images)