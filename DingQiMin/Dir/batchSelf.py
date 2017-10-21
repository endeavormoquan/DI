import os

import gensim
import jieba
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class MySentences(object):
    def __init__(self,dirname):
        self.dirname = dirname

    def __iter__(self):
        for classname in os.listdir(self.dirname):
            for fname in os.listdir(os.path.join(self.dirname,classname)):
                file = open(os.path.join(self.dirname,classname,fname),'r',encoding='utf-8')
                lines = file.readlines()
                if lines.index('$\n'):
                    pos = lines.index('$\n')
                    str = lines[pos+1]
                    text = [word for word in list(jieba.cut(str))]
                    yield text


def createModel():
    sentences = MySentences('D:\QATest')
    model = gensim.models.Word2Vec(sentences,size=28,workers=1,min_count=2)
    model.save('D:\Git\model')


def createVec(dirname,disDirname):
    #  the result will be written to D:\\VecTest in tht format of tfrecords
    model = gensim.models.Word2Vec.load('D:\Git\model')

    classth = 0 # used for label
    for classname in os.listdir(dirname):
        classnum = len(os.listdir(dirname))
        fileth = 0
        for fname in os.listdir(os.path.join(dirname,classname)):
            file = open(os.path.join(dirname,classname,fname),'r',encoding='utf-8')
            lines = file.readlines()
            if lines.index('$\n'):
                pos = lines.index('$\n')
                str = lines[pos + 1]
                from DingQiMin import improveBayes
                rubbishWords = improveBayes.someRubbishWords()
                text = [word for word in list(jieba.cut(str))]
                for word in text:
                    if word in rubbishWords:
                        text.remove(word)
                if len(text) >= 40:
                    tempVec = []
                    Label = [0] * classnum
                    Label[classth] = 1
                    for word in text:
                        if word in model.wv:
                            tempVec.append(model.wv[word])
                    from sklearn import decomposition
                    pca = decomposition.PCA(n_components=28)
                    pca.fit(tempVec)
                    # print(pca.components_.shape)  # word -> vec
                    # print(Label)  # one hot representation

                    #  write to tfrecords
                    fileth += 1
                    vec = pca.components_
                    vec = vec.reshape([28*28])
                    label = np.array(Label)
                    vecfilename = disDirname+'\\vec\\%.5d-of-%.5d.npy' % (classth, fileth)
                    labelfilename = disDirname+'\\label\\%.5d-of-%.5d.npy' % (classth, fileth)
                    np.save(vecfilename,vec)
                    np.save(labelfilename,label)
        classth += 1


def get_batch(dirname,batch_size):
    listToChoose = os.listdir(dirname+'\\vec')
    numOfAll = len(listToChoose)
    ChooseList = np.random.permutation(np.arange(numOfAll))[:batch_size]
    vecBatch = []
    labelBatch = []
    for index in ChooseList:
        filename = dirname+'\\vec\\' + listToChoose[index]
        vecBatch.append(np.load(filename))
    for index in ChooseList:
        filename = dirname+'\\label\\' + listToChoose[index]
        labelBatch.append(np.load(filename))
    return vecBatch,labelBatch


def inception():
    x = tf.placeholder(tf.float32,shape=[None,28*28])
    y_ = tf.placeholder(tf.float32,shape=[None,5])
    x_image = tf.reshape(x,[-1,28,28,1])

    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        biases_initializer=tf.constant_initializer(0.1)):
        with slim.arg_scope([slim.max_pool2d],padding='SAME'):
            branch_1 = slim.conv2d(x_image,32,[1,1])
            branch_1 = slim.max_pool2d(branch_1,[2,2])

            branch_2 = slim.conv2d(x_image,32,[3,3])
            branch_2 = slim.max_pool2d(branch_2,[2,2])

            branch_3 = slim.conv2d(x_image,32,[5,5])
            branch_3 = slim.max_pool2d(branch_3,[2,2])

            branch = tf.concat(axis=3,values=[branch_1,branch_2,branch_3])
            branch = slim.conv2d(branch,32,[3,3])
            branch = slim.max_pool2d(branch,[2,2])

            flat = slim.flatten(branch)
            h_fc1 = slim.fully_connected(flat,1568)

            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = slim.dropout(h_fc1,keep_prob=keep_prob)

            y = slim.fully_connected(h_fc1_drop,5,activation_fn=None)

    cross_entropy = slim.losses.softmax_cross_entropy(y,y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    eval_correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct,tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        vecBatch, labelBatch = get_batch('D:\\VecAndLabelNP',50)
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: vecBatch, y_: labelBatch, keep_prob: 0.5})
            vecBatchForEval, labelBatchForEval = get_batch('D:\\VecAndLabelNPEval',15)
            eval_result = eval_accuracy.eval(feed_dict={
                x: vecBatchForEval, y_: labelBatchForEval, keep_prob: 0.5})
            print("step %d, training and evaluating accuracy %g,%g" % (i, train_accuracy, eval_result))
        train_step.run(feed_dict={x: vecBatch, y_: labelBatch, keep_prob: 0.5})


def CNNInference():
    x = tf.placeholder(tf.float32,[15,28*28])
    y_ = tf.placeholder(tf.float32,shape=[15,5])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable('weight',[5,5,1,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[32],
                                       initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(x_image,conv1_weight,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable('weight',[5,5,32,64],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias',[64],
                                       initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weight,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        fc1_biases = tf.get_variable('bias',[512],
                                     initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weights',[512,5],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        fc2_biases = tf.get_variable('bias',[5],
                                     initializer=tf.constant_initializer(0.1))
        y = tf.matmul(fc1,fc2_weights)+fc2_biases

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    eval_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct, tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        vecBatch, labelBatch = get_batch('D:\\VecAndLabelNP',15)
        if i % 20 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: vecBatch, y_: labelBatch})
            vecBatchForEval, labelBatchForEval = get_batch('D:\\VecAndLabelNPEval',15)
            eval_result = eval_accuracy.eval(feed_dict={
                x: vecBatchForEval, y_: labelBatchForEval})
            print("step %d, training and evaluating accuracy %g,%g" % (i, train_accuracy, eval_result))
        train_step.run(feed_dict={x: vecBatch, y_: labelBatch})



if __name__ == '__main__':
    # createModel()  # only need to run once
    # createVec('D:\\QATest','D:\\VecAndLabelNP')  # only need to run once
    # createVec('D:\\QAEval','D:\\VecAndLabelNPEval')  # only need to run once
    sess = tf.InteractiveSession()
    inception()
    # CNNInference()
