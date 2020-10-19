import torch
import os
import numpy as np
import cv2 as cv
import glob
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from high_to_low import Generator
from high_to_low import data_set

os.environ['cuda_visible_devices'] = '1'


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


def save_img(imgs, img_name, islabel, datadir):
    imgs = np.transpose(imgs, [0, 2, 3, 1])
    if islabel:
        img_dir = './LR_train/' + datadir + '/label/'
    else:
        img_dir = './LR_train/' + datadir + '/sample/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for i in range(len(imgs)):
        img = np.clip((imgs[i] + 1) / 2. * 255., a_min=0., a_max=255.)
        cv.imwrite(img_dir + img_name + '_' + str(i + 1) + '.png', img[:, :, ::-1])
    # print('saved imgs')


def generator(index, datadir):
    model_path = '/home/lipeiying/program/_SR_/Lmser_GAN/high_to_low/checkpoint/model/generator_199.pth'
    dataset = data_set.ImgDataset(index, HR=True)
    loader = DataLoader(dataset, shuffle=True, batch_size=128)
    # generator = Generator.Generator()
    # generator = generator.cuda()
    # a = torch.load(model_path)['model']
    # generator.load_state_dict(a)
    generator = torch.load(model_path)['model']
    generator.eval()

    for i, batch_data in enumerate(loader):
        HR_img, LR_img, z = process(batch_data)
        HR_img = Variable(HR_img, requires_grad=False).cuda()
        LR_img = Variable(LR_img, requires_grad=False).cuda()
        z = Variable(torch.from_numpy(z), requires_grad=False).cuda()

        fake_example, HR_avg_pool = generator(HR_img, z)

        save_img(fake_example.data.cpu().numpy(), str(index) + '_batch' + str(i + 1), islabel=False, datadir=datadir)
        print(i + 1, 'LR saved')
        save_img(HR_img.data.cpu().numpy(), str(index) + '_batch' + str(i + 1), islabel=True, datadir=datadir)
        print(i + 1, 'HR saved')


def get_data(data_dir):
    pic_paths = glob.glob(os.path.join(data_dir, '*.*'))
    pic_paths.sort()
    image_list = []
    for i in range(len(pic_paths)):
        img = np.array(Image.open(pic_paths[i]))

        # print(img.shape)
        image_list.append(img)
        # if i == 0:
        #     print(Image.open(pic_paths[i]))  # mode=RGB size=64x64
    return np.array(image_list)


def img_npy(datadir):
    dir = '/home/lipeiying/program/_SR_/Lmser_GAN/high_to_low/LR_train/' + datadir
    sample_img_path = os.path.join(dir, 'sample')
    label_img_path = os.path.join(dir, 'label')
    sample_paths = glob.glob(os.path.join(sample_img_path, '*.png'))
    sample_paths.sort()
    HR_LR_list = []
    for img_path in sample_paths:
        sample = np.array(Image.open(img_path))
        base_name = os.path.basename(img_path)
        label = np.array(Image.open(os.path.join(label_img_path, base_name)))
        LR_HR = np.array([sample, label])
        HR_LR_list.append(LR_HR)
    res = np.array(HR_LR_list)
    np.save(dir + '.npy', res)


def read_npy():
    dir = '/home/lipeiying/program/_SR_/Lmser_GAN/high_to_low/LR_train/celea_60000_SFD_1'
    npy = np.load(dir + '.npy')
    LR, HR = npy[0][0], npy[0][1]
    cv.imwrite('/home/lipeiying/program/_SR_/Lmser_GAN/high_to_low/LR_train/LR.png', LR)
    cv.imwrite('/home/lipeiying/program/_SR_/Lmser_GAN/high_to_low/LR_train/HR.png', HR)



data_dir = 'all_data'
generator(9, data_dir)
# img_npy(data_dir)

# read_npy()
