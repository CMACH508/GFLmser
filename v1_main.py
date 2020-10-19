import torch
import torch.nn as nn
import numpy as np
import os
import cv2 as cv
import shutil
from os.path import join
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse, sys
import os
import v1_data_set
import v1_discriminator
import v1_lmser

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr_up', type=float, default=0.001)
parser.add_argument('--lr_down', type=float, default=0.001)
parser.add_argument('--lr_dis', type=float, default=0.001)
parser.add_argument('--ckpt_dir', type=str, default='ckpt_2')
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--is_resume', type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
BASE_DIR = sys.path[0]
args.ckpt_dir = join(BASE_DIR, args.ckpt_dir)
if not os.path.exists(args.ckpt_dir):
    os.mkdir(args.ckpt_dir)


def delete_ckpt():
    if os.path.exists(args.ckpt_dir):
        shutil.rmtree(args.ckpt_dir)
        print('===> deleted previous checkpoint !')


def train():
    # delete_ckpt()
    print('===> Build Model')
    lmser_up = v1_lmser.Lmser_up()
    lmser_down = v1_lmser.Lmser_down()
    discriminator = v1_discriminator.Discriminator()
    criterion_mse = nn.MSELoss(reduce=True, size_average=True, reduction='elementwise_mean')

    lmser_up = lmser_up.cuda()
    lmser_down = lmser_down.cuda()
    discriminator = discriminator.cuda()
    criterion_mse = criterion_mse.cuda()

    start_epoch = 1

    if args.is_resume:
        path_dir = ''
        checkpoint_lmser_up = torch.load(join(path_dir, ''))
        checkpoint_lmser_down = torch.load(join(path_dir, ''))
        # checkpoint_dis = torch.load(join(path_dir, ''))
        start_epoch = checkpoint_lmser_up["epoch"] + 1
        lmser_up.load_state_dict(checkpoint_lmser_up["model"].state_dict())
        lmser_down.load_state_dict(checkpoint_lmser_down["model"].state_dict())
        # discriminator.load_state_dict(checkpoint_dis['model'].state_dict())
    print('start epo = ', start_epoch)

    optimizer_up = torch.optim.Adam(lmser_up.parameters(), lr=args.lr_up)
    optimizer_down = torch.optim.Adam(lmser_down.parameters(), lr=args.lr_down)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr_dis)

    print('===> Training')
    real_label = Variable(torch.from_numpy(np.ones([args.batch_size])), requires_grad=False).cuda()
    fake_label = Variable(torch.from_numpy(np.zeros([args.batch_size])), requires_grad=False).cuda()

    HR_img, LR_img, HR_noise_img, fake_lr_img, fake_hr_img = None, None, None, None, None
    g_loss, d_loss, up_loss, down_loss = 0, 0, 0, 0
    for epo in range(start_epoch, 201):
        if epo in [3]:
            optimizer_up.param_groups[0]['lr'] = np.round(optimizer_up.param_groups[0]['lr'] * 0.1, 4)
        if epo in [50, 150]:
            optimizer_down.param_groups[0]['lr'] = np.round(optimizer_down.param_groups[0]['lr'] * 0.1, 4)
        print('new up lr=', optimizer_up.param_groups[0]['lr'])
        print('new down lr=', optimizer_down.param_groups[0]['lr'])
        for npy_index in range(17):
            train_dataset = v1_data_set.TrainDataset(npy_index)
            data_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
            for i, (HR_img, LR_img, HR_noise_img, Dis_data) in enumerate(data_train_loader):

                HR_img = Variable(HR_img, requires_grad=False).cuda()
                HR_noise_img = Variable(HR_noise_img, requires_grad=False).cuda()
                LR_img = Variable(LR_img, requires_grad=False).cuda()
                Dis_data = Variable(Dis_data, requires_grad=False).cuda()

                fake_lr_img, up_list = lmser_up(HR_noise_img)

                if epo in range(3, 49 + 1):  # 3
                    real_out = discriminator(Dis_data).squeeze()
                    fake_out = discriminator(fake_lr_img).squeeze()

                    optimizer_dis.zero_grad()
                    d_real_loss = criterion_mse(real_out.float(), real_label.float())
                    d_fake_loss = criterion_mse(fake_out.float(), fake_label.float())
                    d_loss = (d_real_loss + d_fake_loss) * 0.5
                    d_loss.backward(retain_graph=True)
                    optimizer_dis.step()

                    optimizer_up.zero_grad()
                    g_loss = criterion_mse(fake_out.float(), real_label.float()) * 0.5
                    g_loss.backward(retain_graph=True)
                    optimizer_up.step()

                if epo in range(1, 49 + 1):
                    optimizer_up.zero_grad()
                    up_loss = criterion_mse(fake_lr_img, LR_img)
                    up_loss.backward(retain_graph=False if epo < 50 else True)
                    optimizer_up.step()

                if epo >= 50:  # 10
                    optimizer_down.zero_grad()
                    fake_hr_img = lmser_down(fake_lr_img, up_list)
                    down_loss = criterion_mse(fake_hr_img, HR_img) * 2
                    down_loss.backward()
                    optimizer_down.step()

                if i % 1000 == 0:
                    print(
                        "===> Epoch[{}]({}/{}):g_loss:{:.5f}, D_Loss: {:.5f}, up_Loss:{:.5f}, down_Loss:{:.5f}".format(
                            epo, npy_index + 1, 12, g_loss, d_loss, up_loss, down_loss))

            save_img(fake_lr_img.data.cpu().numpy(), str(epo) + '_lr_fake')
            save_img(LR_img.data.cpu().numpy(), str(epo) + '_lr')
            save_img(HR_img.data.cpu().numpy(), str(epo) + '_hr_label')
            save_img(HR_noise_img.data.cpu().numpy(), str(epo) + '_hr_noise')
            save_checkpoint(lmser_up, epo, 'lmser_up')
            save_checkpoint(discriminator, epo, 'discriminator')
            if epo >= 50:  # 10
                save_checkpoint(lmser_down, epo, 'lmser_down')
                save_img(fake_hr_img.data.cpu().numpy(), str(epo) + '_hr_fake')


def save_img(imgs, img_name):
    # imgs = np.transpose(imgs, [0, 2, 3, 1])
    img_dir = join(args.ckpt_dir, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for i in range(1):
        img = np.clip((imgs[i] + 1) / 2. * 255., a_min=0., a_max=255.).transpose([1, 2, 0])
        # print('img.shape=', img.shape)
        cv.imwrite(join(img_dir, img_name + '.jpg'), img[:, :, ::-1])
    # print('saved imgs')

def save_checkpoint(model, epoch, name):
    model_folder = join(args.ckpt_dir, 'model')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = join(model_folder, "{}-{}.pth".format(name, epoch))
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    train()

   
