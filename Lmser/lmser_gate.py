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

        up6, cor6 = self.corrcoef(out, up6)
        out = self.down_layer_0([out, up6])
        out = self.down_rescale_0(out)

        up5, cor5 = self.corrcoef(out, up5)
        out = self.down_layer_1([out, up5])
        out = self.down_rescale_1(out)

        up4, cor4 = self.corrcoef(out, up4)
        out = self.down_layer_2([out, up4])
        out = self.down_rescale_2(out)

        up3, cor3 = self.corrcoef(out, up3)
        out = self.down_layer_3([out, up3])
        out = self.down_rescale_3(out)

        up2, cor2 = self.corrcoef(out, up2)
        out = self.down_layer_4([out, up2])
        out = self.down_rescale_4(out)

        up1, cor1 = self.corrcoef(out, up1)
        out = self.down_layer_5([out, up1])
        out = self.down_rescale_5(out)

        up0, cor0 = self.corrcoef(out, up0)
        out = self.down_layer_6([out, up0])

        HR_out = self.hr_tanh(self.down_final_process(out))

        # print('LR_out.shape=', LR_out.shape)
        # print('HR_out.shape=', HR_out.shape)
        return LR_out, HR_out, HR_avg_pool, [cor0, cor1, cor2, cor3, cor4, cor5, cor6]
        # return LR_out, HR_out, HR_avg_pool, 0

        # empty_list = np.array(
        #     [np.zeros((128, 64, 64, 64)), np.zeros((128, 64, 32, 32)), np.zeros((128, 64, 16, 16)),
        #      np.zeros((128, 64, 8, 8)),
        #      np.zeros((128, 64, 4, 4)), np.zeros((128, 64, 8, 8)), np.zeros((128, 64, 16, 16))])
        # out, up_list = self.up_layers([out, torch.from_numpy(empty_list)])
        # print('up_list.shape=', up_list.shape)
        # LR_out = self.up_final_process(out)
        # out = self.down_process(LR_out)
        # out = self.down_layers([out, up_list])
        # HR_out = self.down_final_process(out[0])
        # print('LR_out.shape=', LR_out.shape)
        # print('HR_out.shape=', HR_out.shape)
        # return LR_out, HR_out, HR_avg_pool

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
