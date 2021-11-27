#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from opt import Option
from tensorflow.keras.losses import mean_squared_error
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
bone_id = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
           [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
           [13, 17], [17, 18], [18, 19], [19, 20], [20, 21], [20, 23], [21, 22],
           [13, 24], [24, 25], [25, 26], [26, 27], [27, 28], [27, 30], [28, 29]
           ]
parent_id = [0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 20, 21, 13, 24, 25, 26, 27, 27,
             29]
son_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 22, 24, 25, 26, 27, 28, 30, 29]


def losses_init(mask):
    global MASK
    MASK = mask
    print(mask.shape)


def bone_length_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    bone1 = get_single_bone_length(y_true)
    bone2 = get_single_bone_length(y_pred)
    return tf.divide(tf.abs(bone1 - bone2), 120)


def get_single_bone_length(y_true):
    p_idx = tf.constant(parent_id)
    s_idx = tf.constant(son_id)
    p_data = tf.gather(y_true, p_idx, axis=1)
    s_data = tf.gather(y_true, s_idx, axis=1)
    bone_data = tf.square(tf.subtract(p_data, s_data))

    # 3th, 4th; 8th, 9th; 20th-22th; 27th-29th are constant joints (e.g., finger, toe, etc,.)
    d0 = tf.sqrt(bone_data[:, 0, 0] + bone_data[:, 0, 1] + bone_data[:, 0, 2])
    d1 = tf.sqrt(bone_data[:, 1, 0] + bone_data[:, 1, 1] + bone_data[:, 1, 2])
    d2 = tf.sqrt(bone_data[:, 2, 0] + bone_data[:, 2, 1] + bone_data[:, 2, 2])
    # d3  = tf.sqrt(bone_data[:, 3 , 0]+ bone_data[:, 3 , 1]+ bone_data[:, 3 , 2])
    # d4  = tf.sqrt(bone_data[:, 4 , 0]+ bone_data[:, 4 , 1]+ bone_data[:, 4 , 2])
    d5 = tf.sqrt(bone_data[:, 5, 0] + bone_data[:, 5, 1] + bone_data[:, 5, 2])
    d6 = tf.sqrt(bone_data[:, 6, 0] + bone_data[:, 6, 1] + bone_data[:, 6, 2])
    d7 = tf.sqrt(bone_data[:, 7, 0] + bone_data[:, 7, 1] + bone_data[:, 7, 2])
    # d8  = tf.sqrt(bone_data[:, 8 , 0]+ bone_data[:, 8 , 1]+ bone_data[:, 8 , 2])
    # d9  = tf.sqrt(bone_data[:, 9 , 0]+ bone_data[:, 9 , 1]+ bone_data[:, 9 , 2])
    d10 = tf.sqrt(bone_data[:, 10, 0] + bone_data[:, 10, 1] + bone_data[:, 10, 2])
    d11 = tf.sqrt(bone_data[:, 11, 0] + bone_data[:, 11, 1] + bone_data[:, 11, 2])
    d12 = tf.sqrt(bone_data[:, 12, 0] + bone_data[:, 12, 1] + bone_data[:, 12, 2])
    d13 = tf.sqrt(bone_data[:, 13, 0] + bone_data[:, 13, 1] + bone_data[:, 13, 2])
    d14 = tf.sqrt(bone_data[:, 14, 0] + bone_data[:, 14, 1] + bone_data[:, 14, 2])
    d15 = tf.sqrt(bone_data[:, 15, 0] + bone_data[:, 15, 1] + bone_data[:, 15, 2])
    d16 = tf.sqrt(bone_data[:, 16, 0] + bone_data[:, 16, 1] + bone_data[:, 16, 2])
    d17 = tf.sqrt(bone_data[:, 17, 0] + bone_data[:, 17, 1] + bone_data[:, 17, 2])
    d18 = tf.sqrt(bone_data[:, 18, 0] + bone_data[:, 18, 1] + bone_data[:, 18, 2])
    d19 = tf.sqrt(bone_data[:, 19, 0] + bone_data[:, 19, 1] + bone_data[:, 19, 2])
    # d20 = tf.sqrt(bone_data[:, 20, 0]+ bone_data[:, 20, 1]+ bone_data[:, 20, 2])
    # d21 = tf.sqrt(bone_data[:, 21, 0]+ bone_data[:, 21, 1]+ bone_data[:, 21, 2])
    # d22 = tf.sqrt(bone_data[:, 22, 0]+ bone_data[:, 22, 1]+ bone_data[:, 22, 2])
    d23 = tf.sqrt(bone_data[:, 23, 0] + bone_data[:, 23, 1] + bone_data[:, 23, 2])
    d24 = tf.sqrt(bone_data[:, 24, 0] + bone_data[:, 24, 1] + bone_data[:, 24, 2])
    d25 = tf.sqrt(bone_data[:, 25, 0] + bone_data[:, 25, 1] + bone_data[:, 25, 2])
    d26 = tf.sqrt(bone_data[:, 26, 0] + bone_data[:, 26, 1] + bone_data[:, 26, 2])
    # d27 = tf.sqrt(bone_data[:, 27, 0]+ bone_data[:, 27, 1]+ bone_data[:, 27, 2])
    # d28 = tf.sqrt(bone_data[:, 28, 0]+ bone_data[:, 28, 1]+ bone_data[:, 28, 2])
    # d29 = tf.sqrt(bone_data[:, 29, 0]+ bone_data[:, 29, 1]+ bone_data[:, 29, 2])

    # 只保留胳膊，腿部，躯干；移除变化量小的骨头
    res = d0 + d1 + d2 + d5 + d6 + d7 + d10 + d11 + d12 + d13 + d14 + d15 + d16 + d17 + d18 + d19 + d23 + d24 + d25 + d26
    return res


def total_variation_loss(x):
    img_nrows, img_ncols = 32, 32
    assert tf.ndim(x) == 4
    if tf.image_data_format() == 'channels_first':
        a = tf.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                      x[:, :, 1:, :img_ncols - 1])
        b = tf.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                      x[:, :, :img_nrows - 1, 1:])
    else:
        a = tf.square(x[:, :img_nrows - 1, :img_ncols -
                                            1, :] - x[:, 1:, :img_ncols - 1, :])
        b = tf.square(x[:, :img_nrows - 1, :img_ncols -
                                            1, :] - x[:, :img_nrows - 1, 1:, :])
    return tf.sum(tf.pow(a + b, 1.25))


def total_variation_loss(y_true, y_pred):
    tv = tf.abs(
        y_pred[:, :-1, :-1] - y_pred[:, 1:, :-1]
    )
    return tf.math.reduce_sum(tf.math.reduce_sum(tv, axis=[1, 2]))


def final_loss(y_true, y_pred):
    # print(y_true.shape)
    option = Option().parse('walking')
    shape = y_pred.shape
    y_true = y_true * MASK
    y_pred = y_pred * MASK
    y_true = tf.reshape(y_true, [-1, shape[1], shape[2] * shape[3]])
    y_pred = tf.reshape(y_pred, [-1, shape[1], shape[2] * shape[3]])
    return option.mse_loss_weight * mean_squared_error(y_true, y_pred) + \
           option.bone_length_loss_weight * bone_length_loss(y_true, y_pred) + \
           option.tv_loss_weight * total_variation_loss(y_true, y_pred)
