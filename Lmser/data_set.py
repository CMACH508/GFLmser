import numpy as np
# from PIL import Image
# import os, glob
from torch.utils.data import Dataset

HR_data_paths = ['/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/celea_60000_SFD_1.npy',  # (60000)
                 '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/celea_60000_SFD_2.npy',
                 '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/celea_60000_SFD_3.npy',
                 '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/SRtrainset_2.npy',  # (31556)
                 '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/vggcrop_test_lp10.npy',  # (5000)
                 '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/vggcrop_train_lp10_1.npy',
                 '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/vggcrop_train_lp10_2.npy',
                 '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/vggcrop_train_lp10_3.npy',
                 '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/vggcrop_train_lp10_4.npy',
                 '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/HR_64x64/vggcrop_train_lp10_5.npy']  # (86310)

LR_data_paths = ['/home/lipeiying/program/_SR_/Lmser_GAN/dataset/LR_16x16/wider_lnew.npy']


class ImgDataset(Dataset):
    def __init__(self, dataset_index, HR=True):
        super(ImgDataset, self).__init__()
        self.HR_data = get_data(dataset_index, HR)
        self.LR_data = get_data(0, HR=False)

    def __getitem__(self, index):
        return {'HR_img': self.HR_data[index], 'LR_img': self.LR_data[index]}

    def __len__(self):
        return len(self.HR_data)



def get_data(dataset_index, HR=True):
    if HR:
        data = np.load(HR_data_paths[dataset_index])
    else:
        data = np.load(LR_data_paths[dataset_index])
        np.random.shuffle(data)
    imgs = []
    for i in range(len(data)):
        # img = Image.fromarray(data[i])
        # print(type(img))
        imgs.append(data[i])
    imgs = np.array(imgs, dtype=np.float32)
    return imgs
