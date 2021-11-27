#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

from tensorflow.keras.layers import Input, multiply
from tensorflow.keras.optimizers import Adam, Adamax, Nadam
from tensorflow.keras.models import Model

import time
import matplotlib.pyplot as plt
import numpy as np
import sys

# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

from model import Deep_Priors
from adj_matrix import get_cmu_adjacent_matrix
from utils import add_noise, make_noise
from data_processing import DataLoader
from losses import final_loss, losses_init
from opt import Option


def main(option):
    losses = train_model(option)
    plt.plot(list(losses))
    plt.show()
    plt.savefig('./' + 'loss_figure.png')


def train_model(option):
    sigma = 0.
    feature_dimension = 31  # joints num
    input_length = option.motion_range_end - option.motion_range_start
    z = make_noise((input_length, feature_dimension, 3))

    data_loder = DataLoader(option)
    losses_init(data_loder.mask)
    Adjacency = get_cmu_adjacent_matrix(debug=False)
    print("building model...")
    ddp = Deep_Priors(input_length, feature_dimension, Adj=Adjacency, activation='mish')  # activation: relu gelu swish mish
    base_model = ddp.build_model()

    input = base_model.input
    mask_input = Input((input_length, feature_dimension, 3))
    x = base_model.output
    output = multiply([x, mask_input])

    model = Model(inputs=[input, mask_input], outputs=output, name='g_trainer')
    model.compile(optimizer=Adam(), loss=final_loss)
    print("build model done")

    losses = []
    mask = np.expand_dims(data_loder.mask, axis=0)
    std_ct = np.expand_dims(data_loder.std_ct, axis=0)
    print("begin training...")
    early_stopping_condition = 3
    running_duration = 0
    input_noise = add_noise(z, sigma)
    for i in range(option.iteration + 1):
        loss = model.train_on_batch([input_noise, mask], std_ct)
        losses.append(loss)
        print('iter {:0>4d} loss is {:.6f}'.format(i + 1, loss))
    return losses


if __name__ == '__main__':
    option = Option().parse()
    main(option)
