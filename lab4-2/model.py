import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models

class BasicBlock(nn.Module):
    def __init__(self, filter_nums, strides=1, expansion=False):
        super(BasicBlock, self).__init__()
        in_channels, out_channels = filter_nums
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=strides, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        if expansion:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = lambda x:x

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.identity(x)
        out = self.relu(identity + out)
        return out


def BasicBlock_build(filter_nums, block_nums, strides=1, expansion=False):
    build_model = []
    build_model.append(BasicBlock(filter_nums, strides, expansion=expansion))
    filter_nums[0] = filter_nums[1]
    for _ in range(block_nums - 1):
        build_model.append(BasicBlock(filter_nums, strides=1))
    return nn.Sequential(*build_model)


class BottleneckBlock(nn.Module):
    def __init__(self, filter_nums, strides=1, expansion=False):
        super(BottleneckBlock, self).__init__()
        in_channels, mid_channels, out_channels = filter_nums
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, (1, 1), stride=strides, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, (3, 3), stride=1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        if strides!=1 or expansion:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=strides, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = lambda x:x

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        identity = self.identity(x)
        out = self.relu(identity + out)
        return out


def Bottleneck_build(filter_nums, block_nums, strides=1, expansion=False):
    build_model = []
    build_model.append(BottleneckBlock(filter_nums, strides, expansion=expansion))
    filter_nums[0] = filter_nums[2]
    for _ in range(block_nums - 1):
        build_model.append(BottleneckBlock(filter_nums, strides=1))
    return nn.Sequential(*build_model)


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            Bottleneck_build(filter_nums=[64, 64, 256], block_nums=3, strides=1, expansion=True)
        )
        self.conv3 = Bottleneck_build(filter_nums=[256, 128, 512], block_nums=4, strides=2, expansion=True)
        self.conv4 = Bottleneck_build(filter_nums=[512, 256, 1024], block_nums=6, strides=2, expansion=True)
        self.conv5 = Bottleneck_build(filter_nums=[1024, 512, 2048], block_nums=3, strides=2, expansion=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            BasicBlock_build(filter_nums=[64, 64], block_nums=2, strides=1)
        )
        self.conv3 = BasicBlock_build(filter_nums=[64, 128], block_nums=2, strides=2, expansion=True)
        self.conv4 = BasicBlock_build(filter_nums=[128, 256], block_nums=2, strides=2, expansion=True)
        self.conv5 = BasicBlock_build(filter_nums=[256, 512], block_nums=2, strides=2, expansion=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        return out

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary(ResNet50().to(device), (3, 512, 512))
    summary(models.resnet50().to(device), (3, 224, 224))
    summary(ResNet18().to(device), (3, 224, 224))
    summary(models.resnet18().to(device), (3, 224, 224))