import torchvision
from torch import nn


class DeepLabv3Module(nn.Module):
    """
    From the paper https://arxiv.org/abs/1706.05587
    segmentation model using atrous convolution to
    take into account multiscales effect

    n_channel: (int) number of channels of the input image
    """

    def __init__(self, nchannel=3):
        """ """
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DeepLabV3_ResNet101_Weights.DEFAULT"
        )
        # 1 classe !
        self.model.classifier[4] = nn.Conv2d(
            256, 2, kernel_size=(1, 1), stride=(1, 1)
        )

        if nchannel != 3:
            self.model.backbone["conv1"] = nn.Conv2d(
                nchannel,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

    def forward(self, x):
        """ """
        return self.model(x)["out"]
