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
        super(DTPNet, self).__init()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=8, affine=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=16, affine=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding='same')
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(num_features=32, affine=True)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.conv7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same')
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(num_features=64, affine=True)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.conv9 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding='same')
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same')
        self.bn5 = nn.BatchNorm2d(num_features=128, affine=True)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.conv11 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding='same')
        self.bn6 = nn.BatchNorm2d(num_features=256, affine=True)
        self.relu6 = nn.ReLU()

        self.dropout1 = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(256, 1)
        self.regression = nn.Linear(1, 1) 

        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv2(self.conv1(x)))))

        x = self.pool2(self.relu2(self.bn2(self.conv4(self.conv3(x)))))

        x = self.pool3(self.relu3(self.bn3(self.conv6(self.conv5(x)))))

        x = self.pool4(self.relu4(self.bn4(self.conv8(self.conv7(x)))))

        x = self.pool5(self.relu5(self.bn5(self.conv10(self.conv9(x)))))

        x = self.relu6(self.bn6(self.conv11(x)))

        x = self.dropout(x)

        # Applatissement des données pour la FC
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.regression(x)

        return x