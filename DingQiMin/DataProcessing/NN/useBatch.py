import os

import gensim
import jieba
import numpy as np
import tensorflow as tf

CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
MOVING_AVERAGE_DECAY = 0.99


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


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def createModel():
    sentences = MySentences('D:\QATest')
    model = gensim.models.Word2Vec(sentences,size=28,workers=1,min_count=2)
    model.save('D:\Git\model')


def createVec(dirname):
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
                from DingQiMin.DataProcessing.Normal import improveBayes
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
                    vec_raw = pca.components_.tostring()
                    label_raw = np.array(Label).tostring()
                    filename = 'D:\\VecTest\\vecData.tfrecords-%.5d-of-%.5d' % (classth, fileth)
                    writer = tf.python_io.TFRecordWriter(filename)
                    example = tf.train.Example(features = tf.train.Features(feature={
                        'vec':_bytes_feature(vec_raw),
                        # 'label':_int64_feature(np.argmax(Label))
                        'label':_bytes_feature(label_raw)
                    }))
                    writer.write(example.SerializeToString())
                    writer.close()
                    #  end of writing tfrecords
        classth += 1


def batch_vec_label():
    files = tf.train.match_filenames_once('D:\\VecTest\\vecData.tfrecords-*')
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'vec': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }
    )

    decoded_vec = tf.decode_raw(features['vec'], tf.uint8)
    retyped_vec = tf.cast(decoded_vec, tf.float32)
    labels = tf.decode_raw(features['label'], tf.uint8)
    vec = tf.reshape(retyped_vec, [784])
    labels = tf.reshape(labels, [4])  # pay attention


    min_after_dequeue = 100
    batch_size = 5
    capacity = min_after_dequeue+3*batch_size
    vec_batch,label_batch = tf.train.shuffle_batch([vec,labels],
                                                   batch_size=batch_size,
                                                   capacity=capacity,
                                                   min_after_dequeue=min_after_dequeue)

    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    logit = inference(vec_batch,regularizer)
    loss = calc_loss(logit, label_batch)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    with tf.Session() as sess:
        init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(2):
            sess.run(train_step)
            print(i)

        coord.request_stop()
        coord.join(threads)


# def train():
#     with tf.Session() as sess:
#         init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
#         sess.run(init_op)
#
#         vec_batch, label_batch = batch_vec_label()
#
#         regularizer = tf.contrib.layers.l2_regularizer(0.0001)
#         logit = inference(vec_batch, regularizer)
#         loss = calc_loss(logit, label_batch)
#         train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#
#         for i in range(2):
#             sess.run(train_step)
#             print(i)
#
#         coord.request_stop()
#         coord.join(threads)


def inference(input_tensor,regularizer):
    input_tensor = tf.reshape(input_tensor,[-1,28,28,1])
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable('weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor,conv1_weight,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable('weight',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias',[CONV2_DEEP],
                                       initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weight,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[FC_SIZE],
                                     initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)


    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weights',[FC_SIZE,NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[NUM_LABELS],
                                     initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights)+fc2_biases
    return logit


def calc_loss(logit,label):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=tf.argmax(label, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    return loss

if __name__ == '__main__':

    batch_vec_label()