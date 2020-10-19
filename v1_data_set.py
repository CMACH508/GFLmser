import numpy as np
from PIL import Image
import os, glob
from torch.utils.data import Dataset
from os.path import join

names = []

Dis_path = ''
HR_dir = ''
LR_dir = ''
HR_noise_dir = ''


def prepare_data(img):
    img = np.transpose(img, [2, 0, 1])
    img = np.clip(img / 255 * 2 - 1, a_min=-1., a_max=1.)
    return img


class TrainDataset(Dataset):
    def __init__(self, dataset_index):
        super(TrainDataset, self).__init__()
        npy_name = names[dataset_index]
        self.HR_data = get_data(set_path=join(HR_dir, npy_name))
        self.LR_data = get_data(set_path=join(LR_dir, npy_name))
        self.HR_noise_data = get_data(set_path=join(HR_noise_dir, npy_name))
        self.Dis_data = get_data(set_path=Dis_path, dis=True)

    def __getitem__(self, index):
        return prepare_data(self.HR_data[index]), prepare_data(self.LR_data[index]), prepare_data(
            self.HR_noise_data[index]), prepare_data(self.Dis_data[index])

    def __len__(self):
        return len(self.HR_data)


def get_data(set_path, dis=False):
    data = np.load(set_path)
    if dis:
        np.random.shuffle(data)
    imgs = []
    for i in range(len(data)):
        imgs.append(data[i])
    imgs = np.array(imgs, dtype=np.float32)
    return imgs


test_dir = os.path.abspath('./dataset/test_set')
set_dir = 'TestSet'
class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        self.lr_data = get_test_data(join(test_dir, set_dir+'_16x16'))
        self.hr_data = get_test_data(join(test_dir, set_dir+'_64x64'))
        self.noise_data = get_test_data(join(test_dir, set_dir+'_noise_64x64'))
        names = os.listdir(join(test_dir, set_dir+'_64x64'))
        names.sort()
        self.names = np.asarray(names)

    def __getitem__(self, index):
        return prepare_data(self.lr_data[index]), prepare_data(self.hr_data[index]), prepare_data(
            self.noise_data[index])

    def __len__(self):
        return len(self.hr_data)


def get_test_data(path):
    names = os.listdir(path)
    names.sort()
    img_list = []
    for name in names:
        img = np.asarray(Image.open(join(path, name))).astype(np.float32)
        img_list.append(img)
    return np.asarray(img_list)
