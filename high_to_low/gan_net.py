import numpy as np
import tensorflow as tf


class Net(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.in_size = hyper['input_size']
        self.img = tf.placeholder(tf.float32, [None, self.in_size, self.in_size, 3])
        self.z = tf.placeholder(tf.float32, [None, self.in_size, self.in_size, 1])
        self.group = hyper['group']

        self.build_model()

    def generator(self):
        img_noise = tf.concat([self.img, self.z], axis=3)
        out = tf.layers.dense(img_noise, 3, activation=None)
        for group in range(1, self.group + 1):
            if group < 4:  # 1,2,3
                for j in range(1, 3):
                    out = tf.layers.batch_normalization(out)
                    out = tf.nn.relu(out)
                    out = tf.nn.conv2d(out, filter=3, strides=1, padding='SAME', name='group_{}_block_{}'.format(group, j))
                out = tf.nn.avg_pool(out, ksize=3, strides=2)
        print(img_noise.get_shape)

    def discriminator(self, sample):
        pass

    def build_model(self):
        self.generator()
