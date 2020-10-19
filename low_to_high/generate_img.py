import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2 as cv
from low_to_high import data_set

os.environ['cuda_visible_devices'] = '1'


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


def save_img(imgs, img_name, dir):
    # imgs = np.transpose(imgs, [0, 2, 3, 1])
    img_dir = '/home/lipeiying/program/_SR_/Lmser_GAN/low_to_high/result_lmser/' + dir + '/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for i in range(len(imgs)):
        img = np.clip((imgs[i] + 1) / 2. * 255., a_min=0., a_max=255.).transpose([1, 2, 0])
        # print('img.shape=', img.shape)
        cv.imwrite(img_dir + img_name + '_' + str(i + 1) + '.png', img[:, :, :])
    # print('saved imgs')


def generator(index, model_epoch):
    model_path = '/home/lipeiying/program/_SR_/Lmser_GAN/low_to_high/checkpoint_low_high/model/generator_' + model_epoch + '.pth'
    dataset = data_set.ImgDataset(index)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    generator = torch.load(model_path)['model']
    generator.eval()
    for i, batch_data in enumerate(loader):
        sample, label = process(batch_data)

        label = Variable(label, requires_grad=False).cuda()
        sample = Variable(sample, requires_grad=False).cuda()

        fake_sample = generator(sample)

        # dir_name = 'res'+model_epoch
        if i % 7 == 0 or i == 2:
            # print('sample.shape=', sample.shape)
            save_img(sample.data.cpu().numpy(), img_name='batch' + str(i + 1), dir='sample' + model_epoch)
            save_img(fake_sample.data.cpu().numpy(), 'batch' + str(i + 1), 'res' + model_epoch)
            save_img(label.data.cpu().numpy(), 'batch' + str(i + 1), 'label' + model_epoch)

    print('finished')


model_epoch = '199'
generator(0, model_epoch)
