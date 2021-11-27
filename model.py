#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dropout
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.initializers import he_normal, Zeros, glorot_normal, RandomNormal
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from activations import mish, swish, gelu

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class GraphConvolution(layers.Layer):
    def __init__(self, filters, dropout_rate=0.5,
                 use_bias=True, l2_reg=0, seed=1234, name='GCN_layer'):
        super(GraphConvolution, self).__init__()
        self.units = filters
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.seed = seed

    def build(self, input_shape):
        feature_shape = input_shape[0]
        if feature_shape.rank > 3:
            input_dim = int(feature_shape[1] * feature_shape[3])
        else:
            input_dim = int(feature_shape[2])
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      regularizer=l2(self.l2_reg),
                                      initializer=he_normal(seed=self.seed),
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=Zeros(),
                                        name='bias')
        self.built = True

    def call(self, inputs, training=None):
        features, A = inputs
        A = tf.convert_to_tensor(A)
        shape = features.shape
        if shape.rank > 3:
            features = tf.transpose(features, [0, 2, 1, 3])
            features = tf.reshape(
                features, (-1, features.shape[1], features.shape[2] * features.shape[3]))

            output = A @ features
            output = output @ self.kernel
        else:
            output = A @ features
            output = output @ self.kernel

        if self.use_bias:
            output += self.bias
        return output

    def get_config(self):
        config = {'units': self.units,
                  'l2_reg': self.l2_reg,
                  'use_bias': self.use_bias,
                  'seed': self.seed
                  }
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Deep_Priors():
    def __init__(self, input_length, input_dimension, Adj, activation='relu', dropout=0.01, net_depth=4, l2_term=0.00):
        self.motion_length = input_length
        self.motion_dimension = input_dimension
        self.end_channel = self.motion_length * 3
        self.re_term = l2_term
        self.Adj = Adj + np.random.uniform(0, 0.24, size=Adj.shape)
        self.activation = activation
        self.dropout_rate = dropout
        self.depth = net_depth

    def build_model(self, nb_stacks=3, depth_dilation=5):
        filters = [2048, 1024, 512, 256, 128, 128]
        generator_input = Input(
            shape=([self.motion_length, self.motion_dimension, 3]),
            batch_size=None,
            name='input_part')
        # Using GCN encoder-decoder structure
        # input layer
        x0 = self.GraphTemporalConvolution(input=generator_input, adj=self.Adj, filters=filters[0], resi=True, name='0',
                                           act=self.activation)
        # encoder
        x1 = self.GraphTemporalConvolution(input=x0, adj=self.Adj, filters=filters[1], resi=True, name='1',
                                           act=self.activation)
        x2 = self.GraphTemporalConvolution(input=x1, adj=self.Adj, filters=filters[2], resi=True, name='2',
                                           act=self.activation)
        x3 = self.GraphTemporalConvolution(input=x2, adj=self.Adj, filters=filters[3], resi=True, name='3',
                                           act=self.activation)
        x4 = self.GraphTemporalConvolution(input=x3, adj=self.Adj, filters=filters[4], resi=True, name='4',
                                           act=self.activation)
        # latend code, act='mish'
        xy = self.GraphTemporalConvolution(input=x4, adj=self.Adj, filters=filters[5], resi=True, name='5',
                                           act=self.activation)
        # decoder
        y4 = self.GraphTemporalConvolution(input=xy, adj=self.Adj, filters=filters[4], resi=True, name='6',
                                           act=self.activation) + x4
        y3 = self.GraphTemporalConvolution(input=y4, adj=self.Adj, filters=filters[3], resi=True, name='7',
                                           act=self.activation) + x3
        y2 = self.GraphTemporalConvolution(input=y3, adj=self.Adj, filters=filters[2], resi=True, name='8',
                                           act=self.activation) + x2
        y1 = self.GraphTemporalConvolution(input=y2, adj=self.Adj, filters=filters[1], resi=True, name='9',
                                           act=self.activation) + x1
        # output layer
        y0 = self.GraphTemporalConvolution(input=y1, adj=self.Adj, filters=self.end_channel, resi=True, name='10',
                                           act='tanh')
        output = tf.reshape(y0, [-1, self.motion_dimension, 3, self.motion_length])
        output = tf.transpose(output, [0, 3, 1, 2])
        res = tf.add(output, generator_input)  # skip connection

        model = Model(inputs=generator_input, outputs=res)
        return model

    def myprint(s):
        with open('modelsummary.txt', 'w+') as f:
            print(s, file=f)

    def act_result(self, x, name='relu'):
        # using the optimal activation function
        if name == 'relu':
            res = tf.nn.relu(x)
        elif name == 'softmax':
            res = tf.nn.softmax(x)
        elif name == 'mish':
            res = mish(x)
        elif name == 'gelu':
            res = gelu(x)
        elif name == 'swish':
            res = swish(x)
        elif name == 'tanh':
            res = tf.keras.activations.tanh(x)
        return res

    def GraphTemporalConvolution(self, input=None, re_term=0.0001, filters=1024, resi=True, adj=None, T_kernel_size=9,
                                 name='0', act='relu'):
        output = GraphConvolution(filters=filters, l2_reg=self.re_term, name='gcn_{}'.format(name))(
            [input, adj])  # 31*1024
        output = BatchNormalization(axis=1)(output)
        output = self.act_result(output, name=act)

        res = Conv1D(filters=filters, kernel_size=T_kernel_size, padding='same', name='tcn_{}'.format(name),
                     kernel_regularizer=l2(self.re_term))(output)
        res = BatchNormalization(axis=1)(res)
        res = self.act_result(res, name=act)
        res = Dropout(self.dropout_rate)(res)

        if resi:
            return tf.add(output, res)
        else:
            return res
