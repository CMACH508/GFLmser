import numpy as np
import cv2 as cv

names = []


def change_resolution(file):
    # file = ''
    dir = '' + file
    save_path = '' + file
    data_hr = np.load(dir)
    img_list = []
    for i in range(len(data_hr)):
        img = cv.resize(data_hr[i], (16, 16), interpolation=cv.INTER_CUBIC)
        img = cv.resize(img, (64, 64), interpolation=cv.INTER_CUBIC)
        img_list.append(img)
    np.save(save_path, np.asarray(img_list))
    print('Done')


# for j in range(len(names)):
#     change_resolution(names[j])


from os.path import join
import os
from PIL import Image


def testset():
    dir = ''
    save_dir = ''
    test_names = os.listdir(dir)
    test_names.sort()
    for name in test_names:
        hr_img = Image.open(join(dir, name))
        img = cv.resize(hr_img, (16, 16), interpolation=cv.INTER_CUBIC)
        cv.imwrite(join(save_dir, name), img)
    print('Done')


# testset()

def filter():
    new_npy = ''

    dir = ''
    save_dir = ''
    hr_dir = ''
    lr_dir = ''

    celea_hr = ''
    celea_hr1 = ''
    celea_noise_hr = ''
    celea_lr = ''
    cnt = 0
    test_names = os.listdir(save_dir)
    test_names.sort()
    for name in test_names:
        # if '.t7' in name:
        #     continue
        label_img = np.asarray(Image.open(join(save_dir, name)))
        shape = label_img.shape

        if len(shape) == 3 and shape[2] == 3:
            cnt += 1
            noise_img = cv.resize(label_img, (64, 64), interpolation=cv.INTER_CUBIC)
            # noise_img = cv.resize(noise_img, (64, 64), interpolation=cv.INTER_CUBIC)
            cv.imwrite(join(celea_noise_hr, name), noise_img[:, :, ::-1])

            # cv.imwrite(join(celea_hr1, name), label_img[:, :, ::-1])
            # lr_img = cv.resize(hr_img, (16, 16), interpolation=cv.INTER_CUBIC)
            # cv.imwrite(join(celea_lr, name), lr_img[:, :, ::-1])
            #
            # noise_img = cv.resize(hr_img, (32, 32), interpolation=cv.INTER_CUBIC)
            # noise_img = cv.resize(noise_img, (64, 64), interpolation=cv.INTER_CUBIC)
            # cv.imwrite(join(celea_noise_hr, name), noise_img[:, :, ::-1])

            # hr_img = cv.resize(label_img, (64, 64), interpolation=cv.INTER_CUBIC)
            # cv.imwrite(join(hr_dir, name), hr_img[:, :, ::-1])

            # cv.imwrite(join(save_dir, name), label_img[:, :, ::-1])
            # lr_img = cv.resize(label_img, (16, 16), interpolation=cv.INTER_CUBIC)
            # cv.imwrite(join(lr_dir, name), lr_img[:, :, ::-1])
            # noise_img = cv.resize(label_img, (32, 32), interpolation=cv.INTER_CUBIC)
            # noise_img = cv.resize(noise_img, (64, 64), interpolation=cv.INTER_CUBIC)
            # cv.imwrite(join(noise_dir, name), noise_img[: ,:, ::-1])
    print(cnt)


def filter_1():
    label_dir = ''

    hr_npy = ''
    hr_noise_npy = ''
    hr_noise_npy_v2 = ''
    lr_npy = ''

    names = os.listdir(label_dir)
    hr_list, lr_list, noise_list, noise_v2_list = [], [], [], []
    for n in names:
        img_label = np.asarray(Image.open(join(label_dir, n)))
        hr_img = cv.resize(img_label, (64, 64), interpolation=cv.INTER_CUBIC)
        hr_list.append(hr_img)

        lr_img = cv.resize(img_label, (16, 16), interpolation=cv.INTER_CUBIC)
        lr_list.append(lr_img)

        noise_img = cv.resize(img_label, (32, 32), interpolation=cv.INTER_CUBIC)
        noise_img = cv.resize(noise_img, (64, 64), interpolation=cv.INTER_CUBIC)
        noise_list.append(noise_img)

        noise_img_v2 = cv.resize(img_label, (16, 16), interpolation=cv.INTER_CUBIC)
        noise_img_v2 = cv.resize(noise_img_v2, (64, 64), interpolation=cv.INTER_CUBIC)
        noise_v2_list.append(noise_img_v2)

    np.save(hr_npy, np.asarray(hr_list))
    np.save(lr_npy, np.asarray(lr_list))
    np.save(hr_noise_npy, np.asarray(noise_list))
    np.save(hr_noise_npy_v2, np.asarray(noise_v2_list))
    print('Done')


# filter_1()

def test():
    path = ''
    data = np.load(path)
    print(len(data))


# test()
# filter()
import random


def change_name():
    dir = ''
    lr = dir + 'TestSet_16x16'
    hr = dir + 'TestSet_64x64'
    hr_noise = dir + 'TestSet_noise_64x64'
    names = os.listdir(join(dir, lr))
    random.shuffle(names)
    cnt = 0
    for n in names:
        cnt+=1
        img = np.asarray(Image.open(join(lr, n)))
        cv.imwrite(join(lr, str(cnt)+'.jpg'), img[:, :, ::-1])
        os.remove(join(lr, n))

        img = np.asarray(Image.open(join(hr, n)))
        cv.imwrite(join(hr, str(cnt)+'.jpg'), img[:, :, ::-1])
        os.remove(join(hr, n))

        img = np.asarray(Image.open(join(hr_noise, n)))
        cv.imwrite(join(hr_noise, str(cnt)+'.jpg'), img[:, :, ::-1])
        os.remove(join(hr_noise, n))

# change_name()
import numpy as np
from os.path import join
from PIL import Image
import math
import tensorflow as tf
import torch
from torch import nn
sess = tf.Session()

def psnr1(path1, path2):
    img1 = np.asarray(Image.open(path1)).astype(np.double)
    img2 = np.asarray(Image.open(path2)).astype(np.double)
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    # print('psnr1:', 10 * math.log10(255.0 ** 2 / mse))
    psnr = 10 * math.log10(255.0 ** 2 / mse)
    return psnr

def psnr2(path1, path2):
    im1 = tf.image.decode_png(tf.read_file(path1))
    im2 = tf.image.decode_png(tf.read_file(path2))
    psnr = tf.image.psnr(im1, im2, max_val=255)
    # print('psnr2:', sess.run(psnr))
    return sess.run(psnr)


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def psnr3(path1, path2):
    img1 = np.asarray(Image.open(path1))
    img2 = np.asarray(Image.open(path2))
    img1 = img1 / 255.
    img2 = img2 / 255.

    if img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2

    # if im1_in.ndim == 3:
    #     cropped_im1 = im1_in[:, :, :]
    #     cropped_im2 = im2_in[:, :, :]
    # elif im1_in.ndim == 2:
    #     cropped_im1 = im1_in[:, :]
    #     cropped_im2 = im2_in[:, :]
    # else:
    #     raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    img1 = (im1_in * 255).astype(np.float64)
    img2 = (im2_in * 255).astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr

class psnr4(nn.Module):
    def __init__(self, max_val):
        super(psnr4, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        a = torch.tensor(np.asarray(Image.open(a)).astype(np.float32))
        b = torch.tensor(np.asarray(Image.open(b)).astype(np.float32))
        mse = torch.mean((a - b) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return (self.max_val - 10 * torch.log(mse) / self.base10).data

