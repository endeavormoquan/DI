import os

import gensim
import jieba
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from DingQiMin.DataProcessing.Normal import improveBayes

from DingQiMin.DataProcessing.NN import batchSelf


def textCNN():
    x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

