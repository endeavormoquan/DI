import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from DingQiMin.DataProcessing.NN import batchSelf


def textCNN():
    x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        biases_initializer=tf.constant_initializer(0.1)):
        branch1 = slim.conv2d(x_image,32,[9,1],stride=1,padding='VALID')
        branch2 = slim.conv2d(x_image,32,[15,1],stride=1,padding='VALID')
        branch3 = slim.conv2d(x_image,32,[21,1],stride=1,padding='VALID')

        branch1_1 = slim.conv2d(branch1,32,[13,28],stride=1,padding='VALID')
        branch2_1 = slim.conv2d(branch2,32,[7,28],stride=1,padding='VALID')
        branch3_1 = slim.conv2d(branch3,32,[1,28],stride=1,padding="VALID")

        branch = tf.concat(axis=3,values=[branch1_1,branch2_1,branch3_1])
        branch = slim.conv2d(branch,32,[1,1],stride=1,padding='SAME')

        flat = slim.flatten(branch)
        h_fc1 = slim.fully_connected(flat,8*1*32)
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = slim.dropout(h_fc1,keep_prob=keep_prob)

        y = slim.fully_connected(h_fc1_drop,2,activation_fn=None)

    cross_entropy = slim.losses.softmax_cross_entropy(y, y_)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    eval_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct, tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        vecBatch, labelBatch = batchSelf.get_batch('D:\Disease\VecAndLabelNP',50)
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: vecBatch,
                y_: labelBatch,
                keep_prob: 0.5})
            vecBatchForEval, labelBatchForEval = batchSelf.get_batch('D:\Disease\VecAndLabelNPEval',15)
            eval_result = eval_accuracy.eval(feed_dict={
                x: np.array(vecBatchForEval),
                y_: np.array(labelBatchForEval),
                keep_prob: 0.5})
            print("step %d, training and evaluating accuracy %g,%g" % (i, train_accuracy, eval_result))
        train_step.run(feed_dict={x: vecBatch, y_: labelBatch, keep_prob: 0.5})


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    textCNN()
