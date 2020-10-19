import torch
import torch.nn as nn
import numpy as np
import os
import cv2 as cv
import shutil
from torch.autograd import Variable
from torch.utils.data import DataLoader
import hyperparam as Hyper
import data_set
import Generator_lmser as Generator
import Discriminator

# from low_to_high
# from low_to_high
# from low_to_high
# from low_to_high

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
hyper = Hyper.Hyperparameters

hyper['ckpt_dir'] = 'checkpoint_lmser'
hyper['learning_rate'] = 1e-4
hyper['batch_size'] = 128

# def save_img(imgs, img_name, islabel):
#     print(imgs.shape)
#     if islabel:
#         img_dir = './test/label/'
#     else:
#         img_dir = './test/sample/'
#     if not os.path.exists(img_dir):
#         os.makedirs(img_dir)
#     for i in range(len(imgs)):
#         name = img_dir + img_name + str(i + 1) + '.png'
#         print('name=', name)
#         cv.imwrite(name, imgs[i])
#     print('saved imgs')


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


def train(resume=False, paths=None):
    # delete_ckpt()
    print('learning_rate = ', hyper['learning_rate'])
    print('===> Build Model')
    generator = Generator.Generator()
    discriminator = Discriminator.Discriminator()
    criterion_mse = nn.MSELoss(reduce=True, size_average=True, reduction='elementwise_mean')

    # cuda
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_mse = criterion_mse.cuda()

    start_epoch = 0
    print('learning_rate = ', hyper['learning_rate'])
    if resume:
        print("=> loading checkpoint '{}'".format(paths[0]))
        print("=> loading checkpoint '{}'".format(paths[1]))
        checkpoint_G = torch.load(paths[0])
        checkpoint_D = torch.load(paths[1])
        start_epoch = checkpoint_G["epoch"] + 1
        generator.load_state_dict(checkpoint_G["model"].state_dict())
        discriminator.load_state_dict(checkpoint_D['model'].state_dict())

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=hyper['learning_rate'])
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=hyper['learning_rate'])

    print('===> Training')
    real_label = Variable(torch.from_numpy(np.ones([hyper['batch_size']])), requires_grad=False).cuda()
    fake_label = Variable(torch.from_numpy(np.zeros([hyper['batch_size']])), requires_grad=False).cuda()

    generator_log = ''
    discriminator_log = ''
    l2_log = ''
    fake_sample = None
    for epo in range(start_epoch, hyper['epoch'] + 200):
        # if epo != 0 and epo % 50 == 0:
        #     for param_group in g_optimizer.param_groups:
        #         param_group["lr"] = hyper['learning_rate'] * (0.1 ** (epo // 50))
        #     for param_group in d_optimizer.param_groups:
        #         param_group['lr'] = hyper['learning_rate'] * (0.1 ** (epo // 50))
        for npy_index in range(10):
            train_dataset = data_set.ImgDataset(10)
            data_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=hyper['batch_size'], drop_last=True)
            for i, batch_data in enumerate(data_train_loader):
                sample, label = process(batch_data)

                label = Variable(label, requires_grad=False).cuda()
                sample = Variable(sample, requires_grad=False).cuda()
                real_out = discriminator(label).squeeze()

                fake_sample = generator(sample)

                fake_out = discriminator(fake_sample).squeeze()

                d_real_loss = criterion_mse(real_out.float(), real_label.float())
                d_fake_loss = criterion_mse(fake_out.float(), fake_label.float())
                d_loss = d_real_loss + d_fake_loss

                g_loss = criterion_mse(fake_out.float(), real_label.float())

                L2_loss = criterion_mse(fake_sample, label)*10

                generator_log += '\n' + str(g_loss.item())
                discriminator_log += '\n' + str(d_loss.item())
                l2_log += '\n' + str(L2_loss.item())

                d_optimizer.zero_grad()
                g_optimizer.zero_grad()

                d_loss.backward(retain_graph=True)
                g_loss.backward(retain_graph=True)
                L2_loss.backward()

                d_optimizer.step()
                g_optimizer.step()
                # print(sample.shape)
                # print(label.shape)
                # save_img(sample.data.cpu().numpy(), str(i)+'_', islabel=False)
                # save_img(label.data.cpu().numpy(), str(i)+'_', islabel=True)
                if i % 100 == 0:
                    print(
                        "===> Epoch[{}]({}/{}):L2_Loss:{:.5f}, G_Loss: {:.5f}, D_Loss: {:.5f}".format
                        (epo, npy_index + 1, 10, L2_loss, g_loss, d_loss))
                    write_txt(generator_log, 'generator.txt')
                    write_txt(discriminator_log, 'discriminator.txt')
                    write_txt(l2_log, 'l2.txt')

        save_checkpoint(generator, epo, 'generator')
        save_checkpoint(discriminator, epo, 'discriminator')
        # save_img(fake_sample.data.cpu().numpy(), str(epo + 1) + '_' + str(npy_index + 1))
    print('Trainging done')


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
        # print('img.shape=', img.shape)
        cv.imwrite(img_dir + img_name + '.png', img)
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
    model_paths = ['/home/lipeiying/program/_SR_/Lmser_GAN/low_to_high/checkpoint_lmser/model/generator_199.pth',
                   '/home/lipeiying/program/_SR_/Lmser_GAN/low_to_high/checkpoint_lmser/model/discriminator_199.pth']
    # train()
    train(resume=True, paths=model_paths)
