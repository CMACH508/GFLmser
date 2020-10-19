import torch
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import data_set
import hyperparam as Hyper
import model as Model
import Generator
import Discriminator
import shutil
from PIL import Image
import cv2 as cv

os.environ['cuda_visible_devices'] = '1'
hyper = Hyper.Hyperparameters


def delete_ckpt():
    if os.path.exists(hyper['ckpt_dir']):
        shutil.rmtree(hyper['ckpt_dir'])
        print('===> deleted previous checkpoint !')


def main():
    # delete_ckpt()
    train()


def noise_z(shape):
    return np.random.normal(loc=0.0, scale=0.3, size=shape).astype(dtype=np.float32)


def process(batch_data):
    HR_img = batch_data['HR_img']
    LR_img = batch_data['LR_img']
    HR_img = np.transpose(HR_img, [0, 3, 1, 2])
    LR_img = np.transpose(LR_img, [0, 3, 1, 2])
    shape = HR_img.shape
    HR_img = np.clip(HR_img / 255 * 2 - 1, a_min=-1., a_max=1.)
    LR_img = np.clip(LR_img / 255 * 2 - 1, a_min=-1., a_max=1.)
    z = noise_z([shape[0], shape[3]])
    return HR_img, LR_img, z


def train():
    print('===> Build Model ')
    generator = Generator.Generator()
    discriminator = Discriminator.Discriminator()
    criterion_mse = nn.MSELoss(reduce=True, size_average=True, reduction='elementwise_mean')
    # criterion_cross = nn.CrossEntropyLoss()

    # print(model)

    # 放入 cuda
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_mse = criterion_mse.cuda()
    # criterion_cross = criterion_cross.cuda()

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=hyper['learning_rate'])
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=hyper['learning_rate'])
    print("===> Training")
    real_label = Variable(torch.from_numpy(np.ones([hyper['batch_size']])), requires_grad=False).cuda()
    fake_label = Variable(torch.from_numpy(np.zeros([hyper['batch_size']])), requires_grad=False).cuda()
    generator_log = ''
    discriminator_log = ''

    fake_example = None
    # for epo in range(hyper['epoch']):
    for epo in range(hyper['epoch']):
        for data_index in range(10):
            train_set = data_set.ImgDataset(data_index, HR=True)
            train_data_loader = DataLoader(train_set, batch_size=hyper['batch_size'], shuffle=True, drop_last=True)
            for i, batch_data in enumerate(train_data_loader):
                HR_img, LR_img, z = process(batch_data)
                HR_img = Variable(HR_img, requires_grad=False).cuda()
                LR_img = Variable(LR_img, requires_grad=False).cuda()
                z = Variable(torch.from_numpy(z), requires_grad=False).cuda()

                real_out = discriminator(LR_img).squeeze()

                d_real_loss = criterion_mse(real_out.float(), real_label.float())

                fake_example, HR_avg_pool = generator(HR_img, z)
                fake_out = discriminator(fake_example).squeeze()
                d_fake_loss = criterion_mse(fake_out.float(), fake_label.float())
                d_loss = d_real_loss + d_fake_loss

                g_loss = criterion_mse(fake_out.float(), real_label.float())
                L2_loss = criterion_mse(fake_example, HR_avg_pool)

                generator_log += '\n' + str(g_loss.item())
                discriminator_log += '\n' + str(d_loss.item())

                d_optimizer.zero_grad()
                g_optimizer.zero_grad()

                d_loss.backward(retain_graph=True)
                g_loss.backward(retain_graph=True)
                L2_loss.backward()

                d_optimizer.step()
                g_optimizer.step()

                if i % 10 == 0:
                    print(
                        "===> Epoch[{}]({}/{}):L2_Loss:{:.5f}, G_Loss: {:.5f}, D_Loss: {:.5f}".format
                        (epo, data_index + 1, 10, L2_loss, g_loss, d_loss))
                    write_txt(generator_log, 'generator.txt')
                    write_txt(discriminator_log, 'discriminator.txt')

            save_img(fake_example.data.cpu().numpy(), str(epo + 1) + '_' + str(data_index + 1))
            save_checkpoint(generator, epo, 'generator')
            save_checkpoint(discriminator, epo, 'discriminator')


def write_txt(str, file_name):
    dir_path = hyper['ckpt_dir'] + '/log'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    path = os.path.join(dir_path, file_name)
    if not os.path.exists(path):
        f = open(path, 'w')
    else:
        f = open(path, 'a')
    f.write(str)
    f.close()


def save_img(imgs, img_name):
    # imgs = np.transpose(imgs, [0, 2, 3, 1])
    img_dir = hyper['ckpt_dir'] + '/img/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for i in range(1):
        img = np.clip((imgs[i] + 1) / 2. * 255., a_min=0., a_max=255.).transpose([1, 2, 0])
        print('img.shape=', img.shape)
        cv.imwrite(img_dir + img_name + '.png', img[:, :, ::-1])
    print('saved imgs')


def save_checkpoint(model, epoch, name):
    model_folder = hyper['ckpt_dir'] + "/model/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "{}_{}.pth".format(name, epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    main()
