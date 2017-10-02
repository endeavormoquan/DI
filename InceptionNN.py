import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def Inception(input_tensor):
    with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                        stride=1,padding='SAME'):
        # the input_tensor will be the preprocessed tensor
        net = input_tensor
        with tf.variable_scope('mix'):
            with tf.variable_scope('branch_0'):
                branch_0 = slim.conv2d(net,320,[1,1])
            with tf.variable_scope('branch_1'):
                branch_1 = slim.conv2d(net,384,[1,1])
                branch_1 = tf.concat(3,
                                     [slim.conv2d(branch_1,384,[1,3]),slim.conv2d(branch_1,384,[3,1])])
            with tf.variable_scope('branch_2'):
                branch_2 = slim.conv2d(net,448,[1,1])
                branch_2 = slim.conv2d(branch_2,384,[3,3])
                branch_2 = tf.concat(3,
                                     [slim.conv2d(branch_2,384,[1,3]),slim.conv2d(branch_2,384,[3,1])])
            with tf.variable_scope('branch_3'):
                branch_3 = slim.avg_pool2d(net,[3,3])
                branch_3 = slim.conv2d(branch_3,192,[1,1])
            net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
# todo not completed
