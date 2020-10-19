import numpy as np
from PIL import Image
import os, glob
from torch.utils.data import Dataset

data_paths = ['/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/celea_60000_SFD_1.npy',  # (60000)
              '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/celea_60000_SFD_2.npy',
              '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/celea_60000_SFD_3.npy',
              '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/SRtrainset_2.npy',  # (31556)
              '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/vggcrop_test_lp10.npy',  # (5000)
              '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/vggcrop_train_lp10_1.npy',
              '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/vggcrop_train_lp10_2.npy',
              '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/vggcrop_train_lp10_3.npy',
              '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/vggcrop_train_lp10_4.npy',
              '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/LR_train/vggcrop_train_lp10_5.npy',
              '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/lr_hr.npy'
              ]  # (86310)

# high_to_low -> Lmser

class ImgDataset(Dataset):
    def __init__(self, dataset_index):
        super(ImgDataset, self).__init__()
        self.LR_data, self.HR_data = get_data(dataset_index)

    def __getitem__(self, index):
        return {'sample': self.LR_data[index], 'label': self.HR_data[index]}

    def __len__(self):
        return len(self.LR_data)


def get_data(dataset_index):
    # print('dataset path = ', data_paths[dataset_index])
    data = np.load(data_paths[dataset_index])
    np.random.shuffle(data)
    sample_imgs = []
    label_imgs = []
    for i in range(len(data)):
        # img = Image.fromarray(data[i])
        # print(type(img))
        sample_imgs.append(data[i][0])
        label_imgs.append(data[i][1])
    sample_imgs = np.array(sample_imgs, dtype=np.float32)
    label_imgs = np.array(label_imgs, dtype=np.float32)
    return sample_imgs, label_imgs
