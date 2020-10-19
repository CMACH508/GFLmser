import torch
import torch.nn as nn
import numpy as np
import copy

class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, _input):
        out = self.relu1(_input)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out + _input


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
    def __init__(self, up=True, group_n=0):
        super(BasicGroup, self).__init__()
        self.group_n = group_n
        self.up = up

        blocks_list = []
        for i in range(2):
            blocks_list.append(BasicBlock())

        self.layers = nn.Sequential(*blocks_list)

    def forward(self, _input):
        # print(input_list)
        # _input = input_list[0]
        # print('input_list.len=', len(input_list[1]))
        # if self.up:
        # out = self.layers(_input)
        # input_list[1][self.group_n] = out
        # return [out + _input, input_list[1]]
        # else:
        #     p = input_list[1][6-self.group_n]
        # out = self.layers(_input + p)
        # return [out + _input, input_list[0]]
        if self.up:
            out = self.layers(_input)
            return out + _input, out
        else:
            out = self.layers(_input[0] + _input[1])
            return out + _input[0]


class final_process(nn.Module):
    def __init__(self):
        super(final_process, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, input_list):
        _input = input_list[0]
        out = self.conv(_input)
        return [out, input_list[1]]


class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        # self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_down = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.channel_up = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, trans_b):
        batch_size, channel, h, w = trans_b.shape
        for i in range(batch_size):
            batch_b = trans_b[i]
            for j in range(channel):
                channel_b = batch_b[j]
                avg_pool = self.avg_pool(channel_b)


class Lmser(nn.Module):
    def __init__(self):
        super(Lmser, self).__init__()
        self.HR_avg_pool = nn.AvgPool2d(kernel_size=3, stride=4, padding=1)
        self.fully_connect = nn.Linear(in_features=64, out_features=64 * 64)
        self.up_process = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.up_final_process = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.down_process = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.down_final_process = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.lr_tanh = nn.Tanh()
        self.hr_tanh = nn.Tanh()

        self.up_layer_0 = BasicGroup(up=True, group_n=0)
        self.up_layer_1 = BasicGroup(up=True, group_n=1)
        self.up_layer_2 = BasicGroup(up=True, group_n=2)
        self.up_layer_3 = BasicGroup(up=True, group_n=3)
        self.up_layer_4 = BasicGroup(up=True, group_n=4)
        self.up_layer_5 = BasicGroup(up=True, group_n=5)
        self.up_layer_6 = BasicGroup(up=True, group_n=6)
        self.up_rescale_0 = Rescale(down=True)
        self.up_rescale_1 = Rescale(down=True)
        self.up_rescale_2 = Rescale(down=True)
        self.up_rescale_3 = Rescale(down=True)
        self.up_rescale_4 = Rescale(down=False)
        self.up_rescale_5 = Rescale(down=False)

        self.down_layer_0 = BasicGroup(up=False, group_n=0)
        self.down_layer_1 = BasicGroup(up=False, group_n=1)
        self.down_layer_2 = BasicGroup(up=False, group_n=2)
        self.down_layer_3 = BasicGroup(up=False, group_n=3)
        self.down_layer_4 = BasicGroup(up=False, group_n=4)
        self.down_layer_5 = BasicGroup(up=False, group_n=5)
        self.down_layer_6 = BasicGroup(up=False, group_n=6)
        self.down_rescale_0 = Rescale(down=True)
        self.down_rescale_1 = Rescale(down=True)
        self.down_rescale_2 = Rescale(down=False)
        self.down_rescale_3 = Rescale(down=False)
        self.down_rescale_4 = Rescale(down=False)
        self.down_rescale_5 = Rescale(down=False)
        # up_groups_list = []
        # for i in range(4):
        #     up_groups_list.append(BasicGroup(group_n=i))
        #     up_groups_list.append(Rescale(down=False))
        # for j in range(2):
        #     up_groups_list.append(BasicGroup(group_n=j + 4))
        #     up_groups_list.append(Rescale(down=True))
        # up_groups_list.append(BasicGroup(group_n=6))
        # self.up_layers = nn.Sequential(*up_groups_list)
        #
        # down_group_list = []
        # for m in range(2):
        #     down_group_list.append(BasicGroup(up=False, group_n=m))
        #     down_group_list.append(Rescale(down=True))
        # for n in range(4):
        #     down_group_list.append(BasicGroup(up=False, group_n=n + 2))
        #     down_group_list.append(Rescale(down=False))
        # down_group_list.append(BasicGroup(up=False, group_n=6))
        # self.down_layers = nn.Sequential(*down_group_list)

    def forward(self, HR_sample, z_noise):
        HR_avg_pool = self.HR_avg_pool(HR_sample)

        process_z = self.fully_connect(z_noise)
        z = process_z.reshape((-1, 1, 64, 64))
        out = torch.cat([HR_sample, z], dim=1)

        out = self.up_process(out)
        # print(out.shape)

        out, up0 = self.up_layer_0(out)
        out = self.up_rescale_0(out)
        out, up1 = self.up_layer_1(out)
        out = self.up_rescale_1(out)
        out, up2 = self.up_layer_2(out)
        out = self.up_rescale_2(out)
        out, up3 = self.up_layer_3(out)
        out = self.up_rescale_3(out)
        out, up4 = self.up_layer_4(out)
        out = self.up_rescale_4(out)
        out, up5 = self.up_layer_5(out)
        out = self.up_rescale_5(out)
        out, up6 = self.up_layer_6(out)

        LR_out = self.lr_tanh(self.up_final_process(out))

        out = self.down_process(LR_out)

        out = self.down_layer_0([out, up6])
        out = self.down_rescale_0(out)

        out = self.down_layer_1([out, up5])
        out = self.down_rescale_1(out)

        out = self.down_layer_2([out, up4])
        out = self.down_rescale_2(out)

        out = self.down_layer_3([out, up3])
        out = self.down_rescale_3(out)

        out = self.down_layer_4([out, up2])
        out = self.down_rescale_4(out)

        out = self.down_layer_5([out, up1])
        out = self.down_rescale_5(out)

        out = self.down_layer_6([out, up0])

        HR_out = self.hr_tanh(self.down_final_process(out))

        return LR_out, HR_out, HR_avg_pool



