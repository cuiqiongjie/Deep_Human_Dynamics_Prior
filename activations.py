from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

"""Gaussian error linear unit."""
def gelu(x):
  cdf = 0.5 * (1.0 + tf.tanh(
      (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

"""Customized Swish activation."""
def swish(features):
  features = tf.convert_to_tensor(features)
  return features * tf.nn.sigmoid(features)

def mish(features):
  features = tf.convert_to_tensor(features)
  return features * tf.nn.tanh(tf.nn.softplus(features))
