import numpy as np
import os
import glob
from PIL import Image

def img_npy():
    dir = '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/'
    sample_img_path = os.path.join(dir, 'testset')
    label_img_path = os.path.join(dir, 'test_res')
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
    np.save(dir + 'lr_hr.npy', res)

img_npy()
print('done')