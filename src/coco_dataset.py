# This is the correct and complete content for src/coco_dataset.py

import os
import json
import cv2
import numpy as np
from collections import defaultdict

class CocoDataset:
    """
    Custom MindSpore dataset loader for the COCO 2017 object detection format.
    """
    def __init__(self, annotations_file, images_dir):
        print("Initializing COCO dataset loader...")
        self.images_dir = images_dir

        # THIS IS THE PART THAT WAS MISSING
        # It opens the JSON file and loads its content into the 'coco_data' variable.
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # The rest of your code will now work because 'coco_data' exists
        self.images = coco_data['images']
        self.annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)
            
        self.cat_id_to_index = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
        
        print(f"Dataset initialized with {len(self.images)} images.")

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
            
            label_index = self.cat_id_to_index[ann['category_id']]
            labels.append(label_index)

        return (np.array(image, dtype=np.uint8),
                np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int32))

    def __len__(self):
        return len(self.images)