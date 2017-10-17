import tensorflow as tf

def combine():
    files = tf.train.match_filenames_once('D:\\output.tfrecords')
    filename_queue = tf.train.string_input_producer(files,shuffle=False)

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image_raw':tf.FixedLenFeature([],tf.string),
                                           'pixels':tf.FixedLenFeature([],tf.int64),
                                           'label':tf.FixedLenFeature([],tf.int64)
                                       })
    decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
    retyped_images = tf.cast(decoded_images,tf.float32)
    labels = tf.cast(features['label'],tf.int32)
    images = tf.reshape(retyped_images,[784])

    min_after_dequeue = 10000
    batch_size = 100
    capacity = min_after_dequeue+3*batch_size

    image_batch,label_batch = tf.train.shuffle_batch([images,labels],
                                                     batch_size=batch_size,
                                                     capacity=capacity,
                                                     min_after_dequeue=min_after_dequeue)

    return image_batch,label_batch

def inference(input_tensor,weights1,biases1,weights2,biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
    return tf.matmul(layer1,weights2)+biases2

def main(image_batch,label_batch):
    INPUT_NODE = 784
    OUTPUT_NODE = 10
    LAYER1_NODE = 500
    REGULARIZATION_RATE = 0.0001
    TRAINING_STEPS = 5000

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    y = inference(image_batch, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=label_batch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                print("After %d training step(s), loss is %g " % (i, sess.run(loss)))

            sess.run(train_step)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    image_batch, label_batch = combine()
    print(image_batch)
    main(image_batch, label_batch)
