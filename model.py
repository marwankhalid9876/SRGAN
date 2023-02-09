import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import vgg19


from torch import nn


# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         vgg19_model = vgg19(pretrained=True)
#         self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
#
#     def forward(self, img):
#         return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())


    def forward(self, x):
        out1 = self.conv1(x)
        # print('gen1', out1)
        out = self.res_blocks(out1)
        # print('gen2', out)
        out2 = self.conv2(out)
        # print('gen3', out2)
        out = torch.add(out1, out2) #element-wise sum
        # print('gen4', out)
        out = self.upsampling(out)
        # print('gen5', out)
        out = self.conv3(out)
        # print('gen6', out)
        return out

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

# class Discriminator(nn.Module):
#     def __init__(self, input_shape):
#         super(Discriminator, self).__init__()
#
#         self.input_shape = input_shape
#         in_channels, in_height, in_width = self.input_shape
#         patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
#         self.output_shape = (1, patch_h, patch_w)
#
#         def discriminator_block(in_filters, out_filters, first_block=False):
#             layers = []
#             layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
#             if not first_block:
#                 layers.append(nn.BatchNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
#             layers.append(nn.BatchNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         layers = []
#         in_filters = in_channels
#         for i, out_filters in enumerate([64, 128, 256, 512]):
#             layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
#             in_filters = out_filters
#
#         layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
#         layers.append(nn.Sigmoid())
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, img):
#         # print('disc', self.model(img))
#         return self.model(img)