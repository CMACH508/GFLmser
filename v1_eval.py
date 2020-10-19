import torch
import numpy as np
import os
import cv2 as cv
import v1_data_set
from torch.utils.data import DataLoader
import v1_lmser
import argparse
from PIL import Image
import v1_myutils as utils
from os.path import join
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=10)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


save_dir = os.path.abspath('')
def eval():
    cnt = 1
    avg = 0.0
    # model_path = ''
    model_up = ''
    model_down = ''
    dataset = v1_data_set.TestDataset()
    loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
    print('len=', len(loader))
    lmser_up = v1_lmser.Lmser_up()
    lmser_up = torch.load(model_up)['model']
    lmser_up.cuda()
    lmser_up.eval()

    lmser_down = v1_lmser.Lmser_down()
    lmser_down = torch.load(model_down)['model']
    lmser_down.cuda()
    lmser_down.eval()
    batches = 0
    for i, (LR, HR, HR_noise) in enumerate(loader):
        batches += 1
        HR_noise, LR, HR = HR_noise.cuda(), LR.cuda(), HR.cuda()
        LR_out, up_list = lmser_up(HR_noise)
        save_img(LR_out.data.cpu().numpy(), cnt, img_type='_fake_lr', save_dir=save_dir)
        save_img(HR.data.cpu().numpy(), cnt, img_type='_hr', save_dir=save_dir)

        fake_hr = lmser_down(LR, up_list)
        save_img(fake_hr.data.cpu().numpy(), cnt, img_type='_fake_hr', save_dir=save_dir)

        p = get_res(save_dir, cnt)
        avg += p
        cnt += args.batch_size
        if i == 9:
            break
    print('eval done')
    print('total img =', cnt-1)
    print('avg psnr =', avg/batches)


def get_res(save_dir, cnt):
    avg = 0
    for i in range(args.batch_size):
        img_fake_hr = join(save_dir, str(cnt+i) + '')
        img_hr = join(save_dir, str(cnt+i) + '')
        # p = psnr4(img_fake_hr, img_hr)
        p = utils.psnr(img_fake_hr, img_hr)
        avg += p
    return avg/args.batch_size

def save_img(imgs, cnt, img_type, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(len(imgs)):
        img = np.clip((imgs[i] + 1) / 2. * 255., a_min=0., a_max=255.).transpose([1, 2, 0])
        cv.imwrite(join(save_dir, str(cnt+i) + img_type + '.jpg'), img[:, :, ::-1])

eval()
