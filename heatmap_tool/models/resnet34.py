import torch.nn as nn

def conv_block(in_channels, out_channels, activation=False, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels)
    ]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class CustomResNet34(nn.Module):
    def __init__(self, in_channels=3, num_classes=450):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=4),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )
        self.res1 = nn.Sequential(conv_block(64, 64, activation=True), conv_block(64, 64))
        self.res2 = nn.Sequential(conv_block(64, 64, activation=True), conv_block(64, 64))
        self.res3 = nn.Sequential(conv_block(64, 64, activation=True), conv_block(64, 64))
        self.downsample1 = nn.Sequential(conv_block(64, 128, pool=True))
        self.res4 = nn.Sequential(conv_block(64, 128, activation=True, pool=True), conv_block(128, 128))
        self.res5 = nn.Sequential(conv_block(128, 128, activation=True), conv_block(128, 128))
        self.res6 = nn.Sequential(conv_block(128, 128, activation=True), conv_block(128, 128))
        self.res7 = nn.Sequential(conv_block(128, 128, activation=True), conv_block(128, 128))
        self.res8 = nn.Sequential(conv_block(128, 256, activation=True, pool=True), conv_block(256, 256))
        self.downsample2 = nn.Sequential(conv_block(128, 256, pool=True))
        self.res9 = nn.Sequential(conv_block(256, 256, activation=True), conv_block(256, 256))
        self.res10 = nn.Sequential(conv_block(256, 256, activation=True), conv_block(256, 256))
        self.res11 = nn.Sequential(conv_block(256, 256, activation=True), conv_block(256, 256))
        self.res12 = nn.Sequential(conv_block(256, 256, activation=True), conv_block(256, 256))
        self.res13 = nn.Sequential(conv_block(256, 256, activation=True), conv_block(256, 256))
        self.res14 = nn.Sequential(conv_block(256, 512, activation=True, pool=True), conv_block(512, 512))
        self.downsample3 = nn.Sequential(conv_block(256, 512, pool=True))
        self.res15 = nn.Sequential(conv_block(512, 512, activation=True), conv_block(512, 512))
        self.res16 = nn.Sequential(conv_block(512, 512, activation=True), conv_block(512, 512, activation=True))
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.17),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.res2(out) + out
        out = self.res3(out) + out
        out = self.downsample1(out) + self.res4(out)
        out = self.res5(out) + out
        out = self.res6(out) + out
        out = self.res7(out) + out
        out = self.downsample2(out) + self.res8(out)
        out = self.res9(out) + out
        out = self.res10(out) + out
        out = self.res11(out) + out
        out = self.res12(out) + out
        out = self.res13(out) + out
        out = self.downsample3(out) + self.res14(out)
        out = self.res15(out) + out
        out = self.res16(out) + out
        out = self.classifier(out)
        return out
