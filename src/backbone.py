# In src/backbone.py
from mindspore import nn
from mindcv.models import create_model

class EfficientNetBackbone(nn.Cell):
    """
    A custom wrapper around the pre-trained EfficientNet model to expose
    the feature map from its main body, before the final pooling and
    classification layers.
    """
    def __init__(self):
        super().__init__()
        # Load the full pre-trained model once
        full_model = create_model('efficientnet_b1', pretrained=True)
        
        # Extract just the feature-producing part of the model.
        # This is the part that contains all the convolutional blocks.
        self.features = full_model.features

    def construct(self, x):
        # Pass the input through only the feature extraction layers
        return self.features(x)

def create_efficientnet_backbone():
    """
    This function now returns an instance of our custom backbone wrapper.
    """
    print("Loading pre-trained EfficientNet-B1 and creating backbone...")
    backbone = EfficientNetBackbone()
    print("Backbone created successfully!")
    return backbone