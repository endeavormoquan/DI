import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('D:\\MNIST_data',one_hot=True)
sess = tf.InteractiveSession()

def inference():
    x = tf.placeholder(tf.float32,shape=[None,784])
    y_ = tf.placeholder(tf.float32,shape=[None,10])

    x_image = tf.reshape(x,[-1,28,28,1])

    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        biases_initializer=tf.constant_initializer(0.1)):
        with slim.arg_scope([slim.max_pool2d],padding='SAME'):
            h_conv1 = slim.conv2d(x_image,32,[5,5])
            h_pool1 = slim.max_pool2d(h_conv1,[2,2])

            h_conv2 = slim.conv2d(h_pool1,64,[5,5])
            h_pool2 = slim.max_pool2d(h_conv2,[2,2])

            h_pool2_flat = slim.flatten(h_pool2)
            h_fc1 = slim.fully_connected(h_pool2_flat,1024)

            #dropout
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = slim.dropout(h_fc1,keep_prob=keep_prob)

            y_conv = slim.fully_connected(h_fc1_drop,10,activation_fn=None)

    cross_entropy = slim.losses.softmax_cross_entropy(y_conv,y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

if __name__ == '__main__':
    inference()