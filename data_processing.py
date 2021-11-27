#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from vis.regularizers import TotalVariation

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import scipy.io as scio
import sys
import os
import warnings

warnings.filterwarnings('ignore')


class DataLoader(object):
    """
    The Data Loader for human motion recovery .
    """

    def __init__(self,
                 option=None,
                 norm='standard',
                 addnoise=False,
                 noise_degree=0
                 ):
        self.is_add_noise = addnoise
        self.noise_degree = noise_degree

        self.mask = []
        self.gt = []
        self.std_ct = []
        self.std_gt = []

        self.data_name = option.data_dir
        self.save_path = self.data_name.replace('.mat', '/')
        self.option = option

        if option.corruption_method == 'continue_corruption':
            self.missing_marker = [option.missing_joints]
            self.missing_frames = [option.missing_start, option.missing_end]
            self.range = [option.motion_range_start, option.motion_range_end]
        elif option.corruption_method == 'random_corruption':
            self.missing_marker = [option.missing_joints]
            self.missing_ratio = option.missing_ratio
            self.range = [option.motion_range_start, option.motion_range_end]
        elif option.corruption_method == 'gap_corruption':
            self.missing_start = option.missing_start
            self.missing_end = option.missing_end
            self.range = [option.motion_range_start, option.motion_range_end]
        if norm is 'minmax':
            self.norm_method = MinMaxScaler()
        else:
            self.norm_method = StandardScaler()

        self.read_dada()
        self.to_corrupt_motion()

    def read_dada(self):
        print("read data begin")
        data = {}
        try:
            data = scio.loadmat(self.data_name)
        except BaseException:
            print('Can\'t find any motion file form the dir: {}'.format(self.data_name))
        data = data['Coor']
        array = np.array(data[0:][0:])
        data = array[:, self.range[0]:self.range[1]]
        self.gt = np.dstack((data[0::3, :], data[1::3, :], data[2::3, :])).transpose(1, 0, 2)
        print("read data done...")
        return self.gt

    def to_corrupt_motion(self):
        self.mask = np.ones(np.shape(self.gt))

        if self.option.corruption_method == 'continue_corruption':
            begin = self.missing_frames[0]
            end = self.missing_frames[1]
            for i in range(len(self.missing_marker)):
                missing_markers = self.missing_marker[i]
                self.mask[begin:end,
                missing_markers,
                :] = 0
        elif self.option.corruption_method == 'random_corruption':
            np.random.seed(1234)
            zero_frame = np.random.randint(self.range[-1],
                                           size=int((self.range[-1] - self.range[0]) * self.missing_ratio))
            self.mask[zero_frame, self.missing_marker, :] = 0
        elif self.option.corruption_method == 'gap_corruption':
            self.mask[self.missing_start:self.missing_end, :, :] = 0

        self.std_gt = self.normalization(self.gt)
        self.std_gt = np.reshape(self.std_gt, [120, 31, -1])
        self.std_ct = self.std_gt * self.mask

    def normalization(self, x):
        data = np.reshape(x, [self.range[1], -1])
        std_data = self.norm_method.fit_transform(data)
        return std_data

    def denormalization(self, std_x):
        data = np.reshape(std_x, [self.range[1], -1])
        origin_data = self.norm_method.inverse_transform(data)
        return np.reshape(origin_data, [self.range[1], -1, 3])
