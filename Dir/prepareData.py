import gensim
import jieba
import os
import tensorflow as tf
import numpy as np


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
    model = gensim.models.Word2Vec(sentences,size=30,workers=1,min_count=2)
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
                import improveBayes
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
                    pca = decomposition.PCA(n_components=30)
                    pca.fit(tempVec)
                    # print(pca.components_.shape)  # word -> vec
                    # print(Label)  # one hot representation

                    #  write to tfrecords
                    fileth += 1
                    vec_raw = pca.components_.tostring()
                    filename = 'D:\\VecTest\\vecData.tfrecords-%.5d-of-%.5d' % (classth, fileth)
                    writer = tf.python_io.TFRecordWriter(filename)
                    example = tf.train.Example(features = tf.train.Features(feature={
                        'vec':_bytes_feature(vec_raw),
                        'label':_int64_feature(np.argmax(Label))
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
            'label':tf.FixedLenFeature([],tf.int64)
        }
    )
    decoded_vec = tf.decode_raw(features['vec'],tf.uint8)
    retyped_vec = tf.cast(decoded_vec,tf.float32)
    labels = tf.cast(features['label'],tf.int32)
    vec = tf.reshape(retyped_vec,[900])

    min_after_dequeue = 1000
    batch_size = 50
    capacity = min_after_dequeue + 3 * batch_size

    vec_batch,label_batch = tf.train.shuffle_batch([vec,labels],
                                                   batch_size=batch_size,
                                                   capacity=capacity,
                                                   min_after_dequeue=min_after_dequeue)
    return vec_batch,label_batch


if __name__ == '__main__':
    # createModel()
    # createVec('D:\QATest')
    vec_batch,label_batch = batchFiles()
    print(vec_batch)
#todo how to batch the data(pca.components_ and Label)
