import torch
import torch.nn as nn
import hyperparam as Hyper
# from low_to_high
hyper = Hyper.Hyperparameters

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channel)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, _input):
        out = self.bn1(_input)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out


class BasicGroup(nn.Module):
    def __init__(self, n, in_channel, out_channel, islast=False):
        super(BasicGroup, self).__init__()
        self.bi_interpolation = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        blocks_list = []
        self.islast = islast
        for i in range(n):
            blocks_list.append(BasicBlock(in_channel, out_channel))
        self.groups = nn.Sequential(*blocks_list)

    def forward(self, _input):
        out = self.groups(_input)
        if self.islast:
            out = _input + out
        else:
            out = self.bi_interpolation(_input + out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.process = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.final_process = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        groups_list = []
        islast = False
        for i in range(3):
            param_name = 'n_group' + str(i+1)
            if i == 2:
                islast = True
            else:
                islast = False
            groups_list.append(BasicGroup(hyper[param_name], 64, 64, islast))
        self.groups = nn.Sequential(*groups_list)

    def forward(self, _input):
        out = self.process(_input)
        out = self.groups(out)
        out = self.final_process(out)
        return out


