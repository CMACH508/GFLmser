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


class BasicUnit(nn.Module):
    def __init__(self):
        super(BasicUnit, self).__init__()
        blocks_list = []
        for i in range(2):
            blocks_list.append(BasicBlock())
        self.layers = nn.Sequential(*blocks_list)

    def forward(self, _input):
        out = self.layers(_input)
        return out + _input


class BasicGroup(nn.Module):
    def __init__(self, unit_n=2):
        super(BasicGroup, self).__init__()
        # self.process = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.final_process = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        units_list = []
        for i in range(unit_n):
            units_list.append(BasicUnit())

        self.layers = nn.Sequential(*units_list)

    def forward(self, _input):
        out = self.layers(_input)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.process = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.final_process = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.group0 = BasicGroup()
        self.group1 = BasicGroup()
        self.group2 = BasicGroup()
        self.group3 = BasicGroup()
        self.group4 = BasicGroup()
        self.group5 = BasicGroup()
        self.group6 = BasicGroup(unit_n=1)
        self.rescale0 = nn.Sequential(Rescale(down=True))
        self.rescale1 = nn.Sequential(Rescale(down=True))
        self.rescale2 = nn.Sequential(Rescale(down=False))
        self.rescale3 = nn.Sequential(Rescale(down=False))
        self.rescale4 = nn.Sequential(Rescale(down=False))
        self.rescale5 = nn.Sequential(Rescale(down=False))

        # groups_list = []
        # for i in range(2):
        #     groups_list.append(BasicGroup())
        #     groups_list.append(BasicGroup())
        #     groups_list.append(Rescale(down=True))
        # for j in range(4):
        #     groups_list.append(BasicGroup())
        #     groups_list.append(BasicGroup())
        #     groups_list.append(Rescale(down=False))
        # groups_list.append(BasicGroup())
        # self.groups = nn.Sequential(*groups_list)

    def forward(self, _input):
        out = self.process(_input)
        # out = self.groups(out)

        out0 = self.group0(out)
        out = self.rescale0(out0)

        out1 = self.group1(out)
        out = self.rescale1(out1)

        out2 = self.group2(out)
        out = self.rescale2(out2)

        new_out1, _ = self.corrcoef(out, out1)
        out3 = self.group3(out + new_out1)
        out = self.rescale3(out3)

        new_out0, _ = self.corrcoef(out, out0)
        out4 = self.group4(out + new_out0)
        out = self.rescale4(out4)

        out5 = self.group5(out)
        out = self.rescale5(out5)

        out = self.group6(out)

        out = self.final_process(out)
        return out

    def map_corrcoef(self, a, b):
        mean_a = torch.mean(a)
        mean_b = torch.mean(b)
        a = a - mean_a
        b = b - mean_b
        cov = torch.matmul(a, b.transpose(1, 0))
        var_a = torch.matmul(a, a.transpose(1, 0))
        var_b = torch.matmul(b, b.transpose(1, 0))
        corrcoef = cov * torch.rsqrt(var_a * var_b)
        return corrcoef
        # print(corrcoef)

    def corrcoef(self, a, trans_b):
        batch_size, channel, h, w = trans_b.shape
        cor_ = None
        new_b = trans_b.clone()
        for i in range(batch_size):
            batch_b = trans_b[i]
            batch_a = a[i]
            for j in range(channel):
                channel_b = batch_b[j]
                channel_a = batch_a[j]
                channel_b = channel_b.view(1, h * w)
                channel_a = channel_a.view(1, h * w)
                cor = self.map_corrcoef(channel_b, channel_a)
                cor = cor.squeeze()
                if cor > 0:
                    new_b[i][j] = trans_b[i][j].mul(cor)
                if i == 0 and j == 0:
                    cor_ = cor
        return new_b, cor_