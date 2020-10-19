import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, _input):
        out = self.relu1(_input)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out


class Rescale(nn.Module):
    def __init__(self, down=True):
        super(Rescale, self).__init__()
        self.down = down
        self.conv = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.de_conv = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)

    def forward(self, _input):
        if self.down:
            out = self.conv(_input)
        else:
            out = self.de_conv(_input)
        return out


class BasicGroup(nn.Module):
    def __init__(self):
        super(BasicGroup, self).__init__()
        # self.process = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.final_process = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        blocks_list = []
        for i in range(2):
            blocks_list.append(BasicBlock())

        self.layers = nn.Sequential(*blocks_list)

    def forward(self, _input):
        out = self.layers(_input)
        return out + _input


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.process = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.final_process = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        groups_list = []
        for i in range(2):
            groups_list.append(BasicGroup())
            groups_list.append(BasicGroup())
            groups_list.append(Rescale(down=True))
        for j in range(4):
            groups_list.append(BasicGroup())
            groups_list.append(BasicGroup())
            groups_list.append(Rescale(down=False))
        groups_list.append(BasicGroup())
        # groups_list.append(BasicBlock())
        self.groups = nn.Sequential(*groups_list)

    def forward(self, _input):
        out = self.process(_input)
        out = self.groups(out)
        out = self.final_process(out)
        return out


