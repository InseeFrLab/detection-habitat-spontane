import torchvision
from torch import nn
from torchvision.models._api import Weights


class DeepLabv3Module(nn.Module):
    """
    From the paper https://arxiv.org/abs/1706.05587
    segmentation model using atrous convolution to
    take into account multiscales effect

    n_channel: (int) number of channels of the input image
    """

    def __init__(self, nchannel=3, SENTINEL2_RGB_MOCO = False, SENTINEL2_MOCO = False, backbone_requires_grad = True):
        """ """
        super().__init__()

        if SENTINEL2_RGB_MOCO:
            SENTINEL2_RGB_MOCO = Weights(
                url="https://huggingface.co/torchgeo/resnet50_sentinel2_rgb_moco/resolve/main/resnet50_sentinel2_rgb_moco-e3a335e3.pth",  # noqa: E501
                transforms=None,
                meta={
                    "in_chans": nchannel,
                    "model": "resnet50",
                },
            )

            self.model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights_backbone=SENTINEL2_RGB_MOCO
            )
        elif SENTINEL2_MOCO:
            SENTINEL2_MOCO = Weights(
                url="https://huggingface.co/torchgeo/resnet50_sentinel2_all_moco/resolve/main/resnet50_sentinel2_all_moco-e3a335e3.pth",  # noqa: E501
                transforms=None,
                meta={
                    "in_chans": nchannel,
                    "model": "resnet50",
                },
            )

            self.model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights_backbone=SENTINEL2_MOCO
            )
        else:
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                weights="DeepLabV3_ResNet101_Weights.DEFAULT"
            )

        for param in self.model.backbone.parameters():
            param.requires_grad = backbone_requires_grad

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
