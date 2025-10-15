# In src/detection_head.py
from mindspore import nn, ops

class SimpleDetectionHead(nn.Cell):
    """
    A simple detection head that takes a feature map and predicts
    bounding boxes and class scores.
    """
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 9):
        super().__init__()
        
        # A shared convolutional layer to process features
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, pad_mode='pad')
        self.relu = nn.ReLU()

        # 1. The Regressor: Predicts bounding box coordinates (4 values: x, y, width, height)
        self.regressor = nn.Conv2d(256, num_anchors * 4, kernel_size=1)

        # 2. The Classifier: Predicts the class score for each box.
        #    We add +1 to num_classes for the "background" class (no object).
        self.classifier = nn.Conv2d(256, num_anchors * (num_classes + 1), kernel_size=1)

    def construct(self, feature_map):
        # Pass features through the shared layer
        x = self.relu(self.conv(feature_map))

        # Get the box and class predictions
        box_predictions = self.regressor(x)
        class_predictions = self.classifier(x)

        return box_predictions, class_predictions