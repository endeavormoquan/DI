import os

import gensim
import jieba
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from DingQiMin.DataProcessing.Normal import improveBayes


class MySentences(object):
    def __init__(self,dirname):
        self.dirname = dirname

    def __iter__(self):
        for classname in os.listdir(self.dirname):
            print(classname)
            for fname in os.listdir(os.path.join(self.dirname,classname)):
                # print(fname)
                file = open(os.path.join(self.dirname,classname,fname),'r',encoding='utf-8')
                lines = file.readlines()
                if lines.index('$\n'):
                    pos = lines.index('$\n')
                    str = lines[pos+1]
                    text = [word for word in list(jieba.cut(str))]  # todo 此处的分词算法可以替换成自己的
                    yield text


def createModel(srcDir,disDir,vecSize):
    sentences = MySentences(srcDir)
    model = gensim.models.Word2Vec(sentences,size=vecSize,workers=1,min_count=2)
    model.save(disDir)


def createVec(modelDir,srcDir,disDir):

    fileList = os.listdir(disDir+'\\vec')
    for index in fileList:
        os.remove(disDir+'\\vec\\'+index)
    fileList = os.listdir(disDir+'\\label')
    for index in fileList:
        os.remove(disDir+'\\label\\'+index)

    model = gensim.models.Word2Vec.load(modelDir)

    classth = 0 # used for label
    for classname in os.listdir(srcDir):
        classnum = len(os.listdir(srcDir))
        fileth = 0
        for fname in os.listdir(os.path.join(srcDir,classname)):
            file = open(os.path.join(srcDir,classname,fname),'r',encoding='utf-8')
            lines = file.readlines()
            if lines.index('$\n'):
                pos = lines.index('$\n')
                str = lines[pos + 1]

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
                    vec = pca.components_
                    try:
                        vec = vec.reshape([28*28])
                        label = np.array(Label)
                        fileth += 1
                        vecfilename = disDir + '\\vec\\%.5d-of-%.5d.npy' % (classth, fileth)
                        labelfilename = disDir + '\\label\\%.5d-of-%.5d.npy' % (classth, fileth)
                        np.save(vecfilename, vec)
                        np.save(labelfilename, label)
                    except ValueError:
                        print('ValueError')

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

    x = tf.placeholder(tf.float32,shape=[None,28*28],name='x')
    y_ = tf.placeholder(tf.float32,shape=[None,8],name='y_')
    x_image = tf.reshape(x,[-1,28,28,1],name='x_image')

    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        biases_initializer=tf.constant_initializer(0.1)):
        with slim.arg_scope([slim.max_pool2d],padding='SAME'):
            branch_1 = slim.conv2d(x_image,32,[1,1])
            '''
            help(slim.conv2d)
            (inputs, num_outputs, kernel_size, stride=1, padding='SAME', 
            data_format=None, rate=1, activation_fn=<function relu at 0x0000020D55EF7D08>, 
            normalizer_fn=None, normalizer_params=None, 
            weights_initializer=<function variance_scaling_initializer.<locals>._initializer at 0x0000020D5901C8C8>, 
            weights_regularizer=None, 
            biases_initializer=<tensorflow.python.ops.init_ops.Zeros object at 0x0000020D5900C3C8>, 
            biases_regularizer=None, reuse=None, variables_collections=None, 
            outputs_collections=None, trainable=True, scope=None)
            '''
            branch_1 = slim.max_pool2d(branch_1,[2,2])
            '''
            (inputs, kernel_size, stride=2, padding='VALID', 
            data_format='NHWC', outputs_collections=None, scope=None)
            '''

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

            y = slim.fully_connected(h_fc1_drop,8,activation_fn=None)
            b = tf.constant(value = 1,dtype=tf.float32)
            y_eval = tf.multiply(y,b,name='y_eval')

    cross_entropy = slim.losses.softmax_cross_entropy(y,y_)
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    eval_correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct,tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(100):
        vecBatch, labelBatch = get_batch('D:\Departments\VecAndLabelNP',50)
        if i % 10 == 0:
            train_accuracy = sess.run(accuracy,feed_dict={
                x: vecBatch,
                y_: labelBatch,
                keep_prob: 0.5})
            vecBatchForEval, labelBatchForEval = get_batch('D:\Departments\VecAndLabelNPEval',15)
            eval_result = sess.run(eval_accuracy,feed_dict={
                x: vecBatchForEval,
                y_: labelBatchForEval,
                keep_prob: 0.5})
            print("step %d, training and evaluating accuracy %g,%g" % (i, train_accuracy, eval_result))
        sess.run(train_step,feed_dict={x: vecBatch, y_: labelBatch, keep_prob: 0.5})
        # train_step.run(feed_dict={x: vecBatch, y_: labelBatch, keep_prob: 0.5})

    modelPath = 'D:\Departments\cnnModel.ckpt'
    saver.save(sess,modelPath)


def CNNInference():  # 修改eval方法，方法模仿inception函数

    x = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 8], name='y_')
    # x = tf.placeholder(tf.float32,[1,28*28],name='x')
    # y_ = tf.placeholder(tf.float32,shape=[1,8],name='y_')
    x_image = tf.reshape(x, [-1, 28, 28, 1],'x_image')

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
    reshaped = tf.reshape(pool2,[-1,nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        fc1_biases = tf.get_variable('bias',[512],
                                     initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weights',[512,8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        fc2_biases = tf.get_variable('bias',[8],
                                     initializer=tf.constant_initializer(0.1))
        y = tf.matmul(fc1,fc2_weights)+fc2_biases

    b = tf.constant(value=1,dtype=tf.float32)
    logits_eval = tf.multiply(y,b,name='logits_eval')

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    eval_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct, tf.float32))

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        vecBatch, labelBatch = get_batch('D:\Departments\VecAndLabelNP',15)
        if i % 20 == 0:
            train_accuracy = sess.run(accuracy,feed_dict={
                x: vecBatch, y_: labelBatch})
            vecBatchForEval, labelBatchForEval = get_batch('D:\Departments\VecAndLabelNPEval',15)
            eval_result = sess.run(eval_accuracy,feed_dict={
                x: vecBatchForEval, y_: labelBatchForEval})
            print("step %d, training and evaluating accuracy %g,%g" % (i, train_accuracy, eval_result))
        sess.run(train_step,feed_dict={x: vecBatch, y_: labelBatch})
    modelPath = '..\cnnModel.ckpt'
    saver.save(sess,modelPath)
    sess.close()
    # todo should visualize the data


if __name__ == '__main__':
    # createModel('D:\Disease\QATrain','D:\Disease\model',vecSize=28)  # only need to run once
    # createVec('D:\Disease\model','D:\Disease\QATrain','D:\Disease\VecAndLabelNP')
    # createVec('D:\Disease\model','D:\Disease\QAEval','D:\Disease\VecAndLabelNPEval')
    # createModel('D:\Departments\QATrain', 'D:\Departments\model', vecSize=28)  # only need to run once
    # createVec('D:\Departments\model', 'D:\Departments\QATrain', 'D:\Departments\VecAndLabelNP')
    # createVec('D:\Departments\model', 'D:\Departments\QAEval', 'D:\Departments\VecAndLabelNPEval')
    # inception()

    CNNInference()
    