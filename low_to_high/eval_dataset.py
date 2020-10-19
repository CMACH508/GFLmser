import numpy as np
from PIL import Image
import os, glob
from torch.utils.data import Dataset

test_set_path = '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/bicubic_testset.npy'


class TestDataset(Dataset):
    def __init__(self, set_path=test_set_path):
        super(TestDataset, self).__init__()
        self.test_data = get_data(set_path)

    def __getitem__(self, index):
        return self.test_data[index]

    def __len__(self):
        return len(self.test_data)


def get_data(set_path):
    print('set_path=', set_path)
    data = np.load(set_path)
    imgs = []
    for i in range(len(data)):
        # img = Image.fromarray(data[i])
        # print(type(img))
        imgs.append(data[i])
    imgs = np.array(imgs, dtype=np.float32)
    print('data_len=', len(imgs))
    return imgs
