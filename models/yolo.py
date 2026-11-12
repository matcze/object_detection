
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class YOLO(nn.Module):

    def __init__(self, num_classes=3, grid_size=7):
        super().__init__()

        self.num_classes = num_classes
        self.grid_size = grid_size

        # CNN Backbone
        self.backbone = nn.Sequential(

            ConvBlock(3, 32, stride=2),       # 448 -> 224
            ConvBlock(32, 64, stride=2),      # 224 -> 112
            ConvBlock(64, 128, stride=2),     # 112 -> 56
            ConvBlock(128, 256, stride=2),    # 56 -> 28
            ConvBlock(256, 512, stride=2),    # 28 -> 14
            ConvBlock(512, 1024, stride=2),   # 14 -> 7

            ConvBlock(1024, 512),
            ConvBlock(512, 256)
        )

        # Detection Head
        self.head = nn.Conv2d(
            256,
            5 + num_classes,
            kernel_size=1
        )

    def forward(self, x):

        x = self.backbone(x)

        x = self.head(x)

        # Convert:
        # [batch, channels, 7, 7]
        # into:
        # [batch, 7, 7, channels]

        x = x.permute(0, 2, 3, 1)

        return x


if __name__ == "__main__":

    model = YOLO(num_classes=3)

    dummy = torch.randn(2, 3, 448, 448)

    output = model(dummy)

    print("Output shape:", output.shape)

    print("Parameters:",
          sum(p.numel() for p in model.parameters()))
