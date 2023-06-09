import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNModule(nn.Module):
    """
    FasterRCNNModule.
    """

    def __init__(self, nchannel=3):
        """
        Constructor
        """
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT", trainable_backbone_layers=0
        )

        num_classes = 2  # 1 class (building) + background
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass.
        """
        return self.model(x)
