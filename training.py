class DetectionLoss(nn.Cell):
    """
    An improved, but still simplified, loss function for object detection.
    This version attempts to match predictions to ground truth boxes.
    """
    def __init__(self):
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def construct(self, box_preds, class_preds, gt_boxes, gt_labels):
        # Reshape model predictions for easier processing
        box_preds = box_preds.transpose((0, 2, 3, 1)).reshape((-1, 4))
        class_preds = class_preds.transpose((0, 2, 3, 1)).reshape((-1, 9)) # 8 classes + 1 background

        # --- THIS IS THE UPGRADED LOGIC ---
        # For simplicity, we process one image at a time from the batch
        # A full implementation would vectorize this.
        
        batch_size = gt_boxes.shape[0]
        total_loss = 0.0

        for i in range(batch_size):
            # We will perform a very naive matching for demonstration
            # We'll compare the model's first few predictions with the ground-truth boxes
            
            num_gt_objects = ops.count_nonzero(gt_labels[i] > -1) # Count non-padded labels

            if num_gt_objects > 0:
                # Get the predictions and ground truths for this one image
                # For simplicity, we only consider as many predictions as there are GT objects
                preds_to_consider = box_preds[i * num_gt_objects : (i + 1) * num_gt_objects]
                gt_boxes_for_image = gt_boxes[i, :num_gt_objects, :]

                class_preds_to_consider = class_preds[i * num_gt_objects : (i + 1) * num_gt_objects]
                gt_labels_for_image = gt_labels[i, :num_gt_objects]

                # 1. Calculate Regression Loss on the matched pairs
                loss_reg = self.smooth_l1_loss(preds_to_consider, gt_boxes_for_image)

                # 2. Calculate Classification Loss on the matched pairs
                loss_cls = self.cross_entropy_loss(class_preds_to_consider, gt_labels_for_image)
                
                total_loss += loss_reg + loss_cls
            else:
                # If there are no ground truth objects, loss is zero for this image
                total_loss += 0.0
        
        return total_loss / batch_size