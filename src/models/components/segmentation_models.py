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


class DTPNet(nn.Module):
    def __init__(self):
        super(DTPNet, self).__init__()

        self.conv_layers1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1, padding='same'),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=8, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        ])
        
        self.conv_layers2 = nn.ModuleList([
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='same'),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=16, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        ])

        self.conv_layers3 = nn.ModuleList([
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        ])
        
        self.conv_layers4 = nn.ModuleList([
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        ])

        self.conv_layers5 = nn.ModuleList([
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=128, affine=True),
            nn.ReLU()
        ])

        self.conv_layers6 = nn.ModuleList([
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=256, affine=True),
            nn.ReLU()
        ])

        self.dropout1 = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(14*14*256, 1)
        self.regression = nn.Linear(1, 1) 

        
    def forward(self, x):
        for layer in self.conv_layers1:
            x = layer(x)
        for layer in self.conv_layers2:
            x = layer(x)
        for layer in self.conv_layers3:
            x = layer(x)
        for layer in self.conv_layers4:
            x = layer(x)
        for layer in self.conv_layers5:
            x = layer(x)
        for layer in self.conv_layers6:
            x = layer(x)

        x = self.dropout1(x)

        # Applatissement des données pour la FC
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.regression(x)

        return x
