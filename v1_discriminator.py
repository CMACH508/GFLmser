import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.relu_1 = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, stride=1)
        self.relu_2 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, stride=1)

    def forward(self, _input):
        out = self.relu_1(_input)
        out = self.conv_1(out)
        out = self.relu_2(out)
        out = self.conv_2(out)
        return out + _input


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.pre_process = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fully_connect = nn.Linear(in_features=64*4*4, out_features=1)
        self.sigmiod = nn.Sigmoid()
        blocks = []
        in_ch = 64
        out_ch = 64
        for i in range(1, 4+1):
            blocks.append(BasicBlock(in_ch, out_ch))
        for j in range(1, 2+1):
            blocks.append(self.max_pool)
            blocks.append(BasicBlock(in_ch, out_ch))
        self.layers = nn.Sequential(*blocks)

    def forward(self, _input):
        # print('dis input', _input.shape)
        process = self.pre_process(_input)
        out = self.layers(process)
        # print('dis1', out.shape)
        out = self.fully_connect(out.reshape([-1, 64*4*4]))
        # print('dis2', out.shape)
        out_max = self.sigmiod(out)
        return out_max