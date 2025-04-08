import torch.nn as nn
import torchvision


class CustomYOLO(nn.Module):
    def __init__(self, im_size, num_classes):
        super(CustomYOLO, self).__init__()
        self.S = 7  # number of grid cells
        self.B = 5
        self.C = num_classes

        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            *list(resnet18.children())[
                :-2
            ]  # Remove the last two layers (avgpool and fc)
        )
        self.conv_layer = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.fc = self.fc_yolo_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                7 * 7 * 1024,
                2028,
            ),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(2028, self.S * self.S * (5 * self.B + self.C)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_layer(x)
        x = self.fc(x)
        return x
