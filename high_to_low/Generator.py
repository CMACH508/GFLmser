import torch
import torch.nn as nn
# from high_to_low import hyperparam as Hyper
import hyperparam as Hyper

hyper = Hyper.Hyperparameters


class BasicBlock(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, _input):
        out = self.bn1(_input)
        out = self.relu(out)
        out = self.conv1(out)
        return out


class BasicGroup(nn.Module):
    def __init__(self, encoder=True, first=False, last=False, layer=0):
        super(BasicGroup, self).__init__()
        self.encoder = encoder
        blocks_list = []
        in_channel = 64
        out_channel = 64
        for i in range(2):
            if first and i == 0:
                in_channel = 64
            elif last:
                in_channel = 16
                out_channel = 16
            else:
                in_channel = 64
                out_channel = 64
            blocks_list.append(BasicBlock(in_channel, out_channel))
        self.blocks = nn.Sequential(*blocks_list)

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.shuffle_pix = nn.PixelShuffle(upscale_factor=2)

    def forward(self, _input):
        out = self.blocks(_input)
        out = out + _input
        if self.encoder:
            out = self.pool(out)
        else:
            out = self.shuffle_pix(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.pre_precess = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.get_img = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1, stride=1)
        self.fully_connect = nn.Linear(in_features=64, out_features=64 * 64)
        self.HR_avg_pool = nn.AvgPool2d(kernel_size=3, stride=4, padding=1)
        self.tanh = nn.Tanh()
        group_list = []
        n_encoder = hyper['n_encoder']
        n_decoder = hyper['n_decoder']
        for i in range(1, n_encoder+1):
            if i == 1:
                group_list.append(BasicGroup(encoder=True, first=True, layer=i))
            else:
                group_list.append(BasicGroup(encoder=True, first=False, layer=i))
        for j in range(1, n_decoder+1):
            if j == n_decoder:
                group_list.append(BasicGroup(encoder=False, last=True, layer=j))
            else:
                group_list.append(BasicGroup(encoder=False, last=False, layer=j))
        self.groups = nn.Sequential(*group_list)

    def forward(self, HR_sample, z_noise):
        HR_avg_pool = self.HR_avg_pool(HR_sample)

        process_z = self.fully_connect(z_noise)
        z = process_z.reshape((-1, 1, 64, 64))
        out = torch.cat([HR_sample, z], dim=1)

        out = self.pre_precess(out)
        out = self.groups(out)
        out = self.get_img(out)
        tanh = self.tanh(out)
        return tanh, HR_avg_pool