#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@IDE: PyCharm
@author: Cui
@contact: cuiqiongjie@126.com
@time: 2019,7月
Copyright (c),Nanjing University of Science and Technology

@Desc: 

"""
import numpy as np
import tensorflow as tf
import argparse
from data_processing import DataLoader
from tensorflow.keras.layers import Layer
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.append('DeepDynamicPrior')
bone_id = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
           [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
           [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
           [13, 17], [17, 18], [18, 19], [19, 20], [20, 21], [20, 23], [21, 22],
           [13, 24], [24, 25], [25, 26], [26, 27], [27, 28], [27, 30], [28, 29]
           ]


def add_noise(x, sigma):
    np.random.seed(1234)
    noise = np.random.normal(0, sigma, size=x.shape)
    return x + noise


def make_noise(sizes):
    np.random.seed(1234)
    shape = (1, sizes[0], sizes[1], sizes[2])
    noise = np.random.uniform(0, 0.1, size=shape)
    return noise


def bone_length_error(motion, pred):
    total_bone_length_1, total_bone_length_2 = 0., 0.

    for i in range(120):
        total_bone_length_1 += bone_length(motion[i], i, True)
        total_bone_length_2 += bone_length(pred[i])

    bone_length_error = np.abs(total_bone_length_1 - total_bone_length_2)

    return bone_length_error


def bone_length(frame, index=0, isprint=False):
    single_frame_bone_length = 0.

    for i, id in enumerate(bone_id):
        dx = frame[id[0] * 3 + 0] - frame[id[1] * 3 + 0]
        dy = frame[id[0] * 3 + 1] - frame[id[1] * 3 + 1]
        dz = frame[id[0] * 3 + 2] - frame[id[1] * 3 + 2]
        single_frame_bone_length += np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    if isprint:
        print('{}-th frame bone-length is {}'.format(index, single_frame_bone_length))
    return single_frame_bone_length


def mean_squared_error(y_true, y_pred):
    return tf.mean(tf.square(y_pred - y_true), axis=-1)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
