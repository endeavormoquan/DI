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


# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


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


def batchFiles():
    files = tf.train.match_filenames_once('D:\\VecTest\\vecData.tfrecords-*')
    filename_queue = tf.train.string_input_producer(files,shuffle=False)

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'vec':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.string)
        }
    )
    decoded_vec = tf.decode_raw(features['vec'],tf.uint8)
    retyped_vec = tf.cast(decoded_vec,tf.float32)
    labels = tf.decode_raw(features['label'],tf.uint8)
    vec = tf.reshape(retyped_vec,[784])
    labels = tf.reshape(labels,[4])
    # print(vec)  # Tensor("Reshape:0", shape=(784,), dtype=float32)
    return vec,labels


def batch_vec_labels(vec,labels):

    min_after_dequeue = 10
    batch_size = 5
    capacity = min_after_dequeue + 3 * batch_size

    vec_batch,label_batch = tf.train.shuffle_batch([vec,labels],
                                                   batch_size=batch_size,
                                                   capacity=capacity,
                                                   min_after_dequeue=min_after_dequeue)
    return vec_batch,label_batch
    #  Tensor("shuffle_batch:0", shape=(5, 784), dtype=float32)
    #  Tensor("shuffle_batch:1", shape=(5, 4), dtype=uint8)


def inception():
    vec, labels = batchFiles()

    x = tf.placeholder(tf.float32,shape=[None,784])
    y_ = tf.placeholder(tf.float32,shape=[None,4])
    x_image = tf.reshape(x,[-1,28,28,1])

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
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

            branch = tf.concat(axis=3, values=[branch_1, branch_2, branch_3])
            branch = slim.conv2d(branch, 32, [3, 3])
            branch = slim.max_pool2d(branch, [2, 2])

            flat = slim.flatten(branch)
            h_fc1 = slim.fully_connected(flat,1568)

            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = slim.dropout(h_fc1, keep_prob=keep_prob)

            y = slim.fully_connected(h_fc1_drop, 4, activation_fn=None)

    cross_entropy = slim.losses.softmax_cross_entropy(y, y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        vec_batch, label_batch = batch_vec_labels(vec, labels)
        vec_batch, label_batch = sess.run([vec_batch,label_batch])
        train_step.run(feed_dict={x: vec_batch, y_: label_batch, keep_prob: 0.5})
        train_accuracy = accuracy.eval(feed_dict={
            x: vec_batch, y_: label_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

if __name__ == '__main__':
    # createModel()
    # print('model created successfully')
    # createVec('D:\QATest')
    # print('vec created successfully')

    # sess = tf.InteractiveSession()
    # inception()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        vec, labels = batchFiles()
        vec_batch, label_batch = batch_vec_labels(vec, labels)
        print(sess.run(vec_batch),sess.run(label_batch))
        coord.request_stop()
        coord.join(threads)

