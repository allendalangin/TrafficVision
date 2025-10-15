# In src/detector.py
from mindspore import nn
from .backbone import create_efficientnet_backbone
from .detection_head import SimpleDetectionHead

class TrafficVisionDetector(nn.Cell):
    """
    The complete object detection model, combining the backbone and the head.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # 1. Create the pre-trained backbone
        self.backbone = create_efficientnet_backbone()
        
        # 2. Create the detection head
        #    The `in_channels` must match the output channels of the backbone.
        #    For efficientnet_b1, the feature map has 1280 channels.
        self.head = SimpleDetectionHead(in_channels=1280, num_classes=num_classes)

    def construct(self, image):
        # Pass the image through the backbone to get high-level features
        features = self.backbone(image)
        
        # Pass the features through the head to get the final predictions
        box_preds, class_preds = self.head(features)
        
        return box_preds, class_preds