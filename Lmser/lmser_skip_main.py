import torch
import torch.nn as nn
import numpy as np
import os
import cv2 as cv
import shutil
from torch.autograd import Variable
from torch.utils.data import DataLoader

import data_set
import hyperparam as Hyper
import lmser_no_skip
import Discriminator

# from Lmser import data_set
# from Lmser import hyperparam as Hyper
# from Lmser import lmser_gate
# from Lmser import Discriminator




def process(batch_data):
    HR_img = batch_data['label']
    LR_img = batch_data['sample']
    HR_img = np.transpose(HR_img, [0, 3, 1, 2])
    LR_img = np.transpose(LR_img, [0, 3, 1, 2])
    HR_img = np.clip(HR_img / 255 * 2 - 1, a_min=-1., a_max=1.)
    LR_img = np.clip(LR_img / 255 * 2 - 1, a_min=-1., a_max=1.)
    # print('LR_shape=', LR_img.shape)
    # print('HR.shape=', HR_img.shape)
    return LR_img, HR_img


def delete_ckpt():
    if os.path.exists(hyper['ckpt_dir']):
        shutil.rmtree(hyper['ckpt_dir'])
        print('===> deleted previous checkpoint !')


def noise_z(shape):
    return np.random.normal(loc=0.0, scale=1.0, size=shape).astype(dtype=np.float32)


def prepare_data(batch_data):
    HR_img = batch_data['HR_img']
    LR_img = batch_data['LR_img']
    HR_img = np.transpose(HR_img, [0, 3, 1, 2])
    LR_img = np.transpose(LR_img, [0, 3, 1, 2])
    shape = HR_img.shape
    HR_img = np.clip(HR_img / 255 * 2 - 1, a_min=-1., a_max=1.)
    LR_img = np.clip(LR_img / 255 * 2 - 1, a_min=-1., a_max=1.)
    z = noise_z([shape[0], shape[3]])
    return HR_img, LR_img, z


def train(cuda='1', resume=False, paths=None, ckpt='ckpt'):
    # delete_ckpt()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    hyper['ckpt_dir'] = ckpt
    print('===> Build Model')
    # generator = Generator.Generator()
    lmser = lmser_no_skip.Lmser()
    discriminator = Discriminator.Discriminator()
    criterion_mse = nn.MSELoss(reduce=True, size_average=True, reduction='elementwise_mean')

    # cuda
    # generator = generator.cuda()
    lmser = lmser.cuda()
    discriminator = discriminator.cuda()
    criterion_mse = criterion_mse.cuda()

    start_epoch = 1

    if resume:
        print("===> loading checkpoint '{}'".format(paths[0]))
        print("===> loading checkpoint '{}'".format(paths[1]))
        checkpoint_lmser = torch.load(paths[0])
        checkpoint_D = torch.load(paths[1])
        start_epoch = checkpoint_lmser["epoch"] + 1
        lmser.load_state_dict(checkpoint_lmser["model"].state_dict())
        discriminator.load_state_dict(checkpoint_D['model'].state_dict())

    lmser_optimizer = torch.optim.Adam(lmser.parameters(), lr=hyper['learning_rate'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=hyper['learning_rate'])

    print('===> Training')
    real_label = Variable(torch.from_numpy(np.ones([hyper['batch_size']])), requires_grad=False).cuda()
    fake_label = Variable(torch.from_numpy(np.zeros([hyper['batch_size']])), requires_grad=False).cuda()

    lmser_log = ''
    generator_log = ''
    discriminator_log = ''
    l2_log = ''
    lr_img = None
    hr_img = None
    HR_img = None
    print('learning_rate = ', hyper['learning_rate'])
    for epo in range(start_epoch, hyper['epoch'] - start_epoch + 1):
        if epo != 0 and epo % 50 == 0:
            for param_group in lmser_optimizer.param_groups:
                param_group["lr"] = hyper['learning_rate'] * (0.1 ** (epo // 50))
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = hyper['learning_rate'] * (0.1 ** (epo // 50))
            print('learning_rate = ', hyper['learning_rate'] * (0.1 ** (epo // 50)))
        for npy_index in range(10):
            train_dataset = data_set.ImgDataset(npy_index)
            data_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=hyper['batch_size'], drop_last=True)
            for i, batch_data in enumerate(data_train_loader):
                HR_img, LR_img, z = prepare_data(batch_data)
                HR_img = Variable(HR_img, requires_grad=False).cuda()
                LR_img = Variable(LR_img, requires_grad=False).cuda()
                z = Variable(torch.from_numpy(z), requires_grad=False).cuda()

                real_out = discriminator(LR_img).squeeze()

                lr_img, hr_img, HR_avg_pool = lmser(HR_img, z)
                fake_out = discriminator(lr_img).squeeze()

                d_real_loss = criterion_mse(real_out.float(), real_label.float())
                d_fake_loss = criterion_mse(fake_out.float(), fake_label.float())
                d_loss = d_real_loss + d_fake_loss

                g_loss = criterion_mse(fake_out.float(), real_label.float())
                L2_loss = criterion_mse(lr_img, HR_avg_pool) * 5
                lmser_loss = criterion_mse(hr_img, HR_img) * 0.1

                d_optimizer.zero_grad()
                lmser_optimizer.zero_grad()

                d_loss.backward(retain_graph=True)
                g_loss.backward(retain_graph=True)
                L2_loss.backward(retain_graph=True)
                lmser_loss.backward()

                # d_loss.backward(retain_graph=True)
                # g_loss.backward(retain_graph=True)
                # L2_loss.backward()

                d_optimizer.step()
                lmser_optimizer.step()

                if i % 100 == 0:
                    lmser_log = lmser_log + '\n' + str(lmser_loss.item())
                    discriminator_log = discriminator_log + '\n' + str(d_loss.item())
                    l2_log = l2_log + '\n' + str(L2_loss.item())

                if i % 1000 == 0:
                    print(
                        "===> Epoch[{}]({}/{}):g_loss:{:.5f}, L2_Loss:{:.5f}, D_Loss: {:.5f}, lmser_Loss: {:.5f}".format
                        (epo, npy_index + 1, 10, g_loss, L2_loss, d_loss, lmser_loss))
                    lmser_log = write_txt(lmser_log, 'lmser.txt')
                    discriminator_log = write_txt(discriminator_log, 'discriminator.txt')
                    l2_log = write_txt(l2_log, 'l2.txt')

            save_img(lr_img.data.cpu().numpy(), str(epo + 1) + '_lr')
            save_img(hr_img.data.cpu().numpy(), str(epo + 1) + '_hr')
            save_img(HR_img.data.cpu().numpy(), str(epo + 1) + '_hr_label')
            save_checkpoint(lmser, epo+1, 'lmser')
            save_checkpoint(discriminator, epo+1, 'discriminator')


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
    return ''


def save_img(imgs, img_name):
    # imgs = np.transpose(imgs, [0, 2, 3, 1])
    img_dir = hyper['ckpt_dir'] + '/img/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for i in range(1):
        img = np.clip((imgs[i] + 1) / 2. * 255., a_min=0., a_max=255.).transpose([1, 2, 0])
        # print('img.shape=', img.shape)
        cv.imwrite(img_dir + img_name + '.png', img[:, :, ::-1])
    # print('saved imgs')


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
    hyper = Hyper.Hyperparameters
    hyper['batch_size'] = 256
    hyper['learning_rate'] = 1e-4

    model_paths = ['/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/pre_trained/pre_lmser_11.pth',
                   '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/pre_trained/pre_discriminator_11.pth']
    train(cuda='1', resume=True, paths=model_paths, ckpt='ckpt_skip_pre')

    # train(cuda='0', ckpt='ckpt_skip')
