import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import cv2 as cv
# from low_to_high import data_set
import eval_dataset
os.environ['cuda_visible_devices'] = '1'


def process(batch_data_lr):
    batch_data_lr = np.transpose(batch_data_lr, [0, 3, 1, 2])
    batch_data_lr = np.clip(batch_data_lr / 255 * 2 - 1, a_min=-1., a_max=1.)
    return batch_data_lr


def save_img(imgs, img_name, save_dir):
    imgs = np.transpose(imgs, [0, 2, 3, 1])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(len(imgs)):
        img = np.clip((imgs[i] + 1) / 2. * 255., a_min=0., a_max=255.)
        name = img_name + '_' + str(i + 1) + '.png'
        path = os.path.join(save_dir, name)
        cv.imwrite(path, img[:, :, ::-1])
    # print('saved imgs')


def generator():
    save_dir = '/home/lipeiying/program/_SR_/Lmser_GAN/eval/l2h_lr_hr'
    model_path = '/home/lipeiying/program/_SR_/Lmser_GAN/low_to_high/checkpoint_low_high/model/generator_399.pth'

    # save_dir = '/home/lipeiying/program/_SR_/Lmser_GAN/eval/l2h_lr_hr_lmser'
    # model_path = '/home/lipeiying/program/_SR_/Lmser_GAN/low_to_high/checkpoint_lmser/model/generator_399.pth'

    dataset = eval_dataset.TestDataset(set_path='/home/lipeiying/program/_SR_/Lmser_GAN/dataset/testset.npy')
    loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    generator = torch.load(model_path)['model']
    generator.eval()

    for i, batch_data in enumerate(loader):
        print(i + 1, ' batch')
        sample = process(batch_data)

        sample = Variable(sample, requires_grad=False).cuda()

        fake_sample = generator(sample)

        save_img(sample.data.cpu().numpy(), 'lr_batch' + str(i + 1), save_dir)
        save_img(fake_sample.data.cpu().numpy(), 'hr_batch' + str(i + 1), save_dir)

    print('finished')


generator()



