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

        """
        # Anchor generation
        anchor_generator = AnchorGenerator(
            sizes=((10,), (20,), (30,), (40,), (50,), (60,))
            aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)])
        )
        model.rpn.anchor_generator = anchor_generator

        # 256 because that's the number of features that FPN returns
        model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
        """

    def forward(self, x):
        """
        Forward pass.
        """
        return self.model(x)
