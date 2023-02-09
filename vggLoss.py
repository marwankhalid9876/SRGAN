import torch.nn as nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        blocks.append(vgg19(pretrained=True).to('cuda').features[:4].eval())
        blocks.append(vgg19(pretrained=True).to('cuda').features[4:9].eval())
        blocks.append(vgg19(pretrained=True).to('cuda').features[9:18].eval())
        blocks.append(vgg19(pretrained=True).to('cuda').features[18:27].eval())
        blocks.append(vgg19(pretrained=True).to('cuda').features[27:36].eval())
        blocks = nn.ModuleList(blocks)
        self.blocks = blocks
        self.loss = nn.MSELoss()

        for param in self.blocks.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        for block in self.blocks:
            input = block(input)
            target = block(target)

        return self.loss(input, target,)
