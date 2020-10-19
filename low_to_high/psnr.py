import math
import numpy as np
import tensorflow as tf

# def PSNR(pred, gt, shave_border=0):
#     height, width = pred.shape[:2]
#     pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
#     gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
#     imdff = pred - gt
#     rmse = math.sqrt(np.mean(imdff ** 2))
#     if rmse == 0:
#         return 100
#     return 20 * math.log10(255.0 / rmse)

# def psnr(im1, im2):
#     im1 = tf.decode_png('path/to/im1.png')
#     im2 = tf.decode_png('path/to/im2.png')
#     # Compute PSNR over tf.uint8 Tensors.
#     psnr1 = tf.image.psnr(im1, im2, max_val=255)
import glob
import os
sess = tf.Session()
path_label = '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/LS3D_100_label'
path_res = '/home/lipeiying/program/_SR_/Lmser_GAN/dataset/LS3D_100_no_skip'
res_names = os.listdir(path_res)
avg_psnr = 0
for i in range(len(res_names)):
    res_path = os.path.join(path_res, res_names[i])
    label_path = os.path.join(path_label, res_names[i])
    # print('path=', label_path)
    im1 = tf.image.decode_png(tf.read_file(res_path))
    im2 = tf.image.decode_jpeg(tf.read_file(label_path))
    psnr = tf.image.psnr(im1, im2, max_val=255)
    # print(sess.run(psnr))
    avg_psnr += psnr
print(len(res_names), 'imgs')
print(sess.run(avg_psnr)/len(res_names))
# print('avg_psnr = ', sess.run(avg_psnr)/len(res_names))


