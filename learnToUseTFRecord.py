import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def store():
    mnist = input_data.read_data_sets('D:\\MNIST_data',dtype=tf.uint8,one_hot=True)
    images = mnist.train.images
    labels = mnist.train.labels
    pixels = images.shape[1]
    print(images.shape) # (55000,784)
    num_examples = mnist.train.num_examples

    filename = 'D:\\output.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        image_raw = images[index].tostring()
        '''
        message Example {
            Feature features = 1;
        };
        
        message Features {
            map{string ,Feature> feature = 1;
        };
        
        message Feature {
            oneof kind{
                BytesList bytes_list = 1;
                FloatList float_list = 2;
                Int64List int64_list = 3;
            }
        };
        '''
        example = tf.train.Example(features=tf.train.Features(feature={
            'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)
        }))

        writer.write(example.SerializeToString())
    writer.close()

def unstore():
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(['D:\\output.tfrecords'])

    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'pixels':tf.FixedLenFeature([],tf.int64),
            'label':tf.FixedLenFeature([],tf.int64)
        })
    images = tf.decode_raw(features['image_raw'],tf.uint8)
    labels = tf.cast(features['label'],tf.int32)
    pixels = tf.cast(features['pixels'],tf.int32)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(10):
        image,label,pixel = sess.run([images,labels,pixels])
        print(label,pixel)

def tensorflowQueue():
    q = tf.FIFOQueue(2,'int32')
    init = q.enqueue_many(([0,10],),name=None)
    x = q.dequeue()
    y = x+1
    q_inc = q.enqueue([y])

    with tf.Session() as sess:
        init.run()
        for index in range(5):
            v,index = sess.run([x,q_inc])
            print(v)

def exams():
    num_shards = 2
    instances_per_shard = 2
    for i in range(num_shards):
        filename = 'D:\examsData\\data.tfrecords-%.5d-of-%.5d'%(i,num_shards)
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(instances_per_shard):
            example = tf.train.Example(features = tf.train.Features(feature={
                'i':_int64_feature(i),
                'j':_int64_feature(j)
            }))
            writer.write(example.SerializeToString())
        writer.close()
    print('exams done')

def readData():
    files = tf.train.match_filenames_once("D:\examsData\\data.tfrecords-*")
    filename_queue = tf.train.string_input_producer(files,shuffle=False)
    init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'i':tf.FixedLenFeature([],tf.int64),
            'j':tf.FixedLenFeature([],tf.int64),
        })

    example,label = features['i'],features['j']

    batch_size = 3
    capacity = 100+3*batch_size

    example_batch,label_batch = tf.train.batch([example,label],batch_size=batch_size,capacity=capacity)

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(2):
            cur_example_batch,cur_label_batch = sess.run([example_batch,label_batch])
            print(cur_example_batch,cur_label_batch)

        coord.request_stop()
        coord.join(threads)
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     print(sess.run(files))
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     for i in range(6):
    #         print(sess.run([features['i'], features['j']]))
    #     coord.request_stop()
    #     coord.join(threads)

if __name__ == '__main__':
    # store()
    # unstore()
    # tensorflowQueue()
    # exams()
    readData()
