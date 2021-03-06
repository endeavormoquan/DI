import tensorflow as tf
import os
import numpy as np
import gensim
import jieba
from tensorflow.contrib import rnn
from DingQiMin.DataProcessing.NN.batchSelf import createModel
from DingQiMin.DataProcessing.Normal import improveBayes

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
                        vec = vec.reshape([28*40])
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
        temp = np.load(filename)
        vecBatch.append(temp.tolist())
    vecBatch = np.array(vecBatch)
    for index in ChooseList:
        filename = dirname+'\\label\\' + listToChoose[index]
        temp = np.load(filename).tolist()
        labelBatch.append(temp)
    labelBatch = np.array(labelBatch)
    return vecBatch,labelBatch


def RNN():
    learning_rate = 0.001
    training_steps = 10000
    batch_size = 128
    display_step = 200

    num_input = 28  # MNIST data input (img shape: 28*28)
    timesteps = 40  # timesteps
    num_hidden = 128  # hidden layer num of features
    num_classes = 2  # MNIST total classes (0-9 digits)

    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(X, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    eval_correct = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct,tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, training_steps+1):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)

            batch_x, batch_y = get_batch('D:\Disease\VecAndLabelNP', batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)



            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:

                batch_x_foreval, batch_y_foreval = get_batch('D:\Disease\VecAndLabelNPEval', 15)
                batch_x_foreval = batch_x_foreval.reshape((15, timesteps, num_input))
                eval_result = eval_accuracy.eval(feed_dict={
                    X: batch_x_foreval,
                    Y: batch_y_foreval})
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc)+", eval Accuracy= " + \
                      "{:.3f}".format(eval_result))


if __name__ == '__main__':
    createModel('D:\Departments\QATrain','D:\Departments\model',vecSize=40)  # only need to run once
    print('model created')
    createVec('D:\Departments\model','D:\Departments\QATrain','D:\Departments\VecAndLabelNP')
    print('vec created')
    createVec('D:\Departments\model','D:\Departments\QAEval','D:\Departments\VecAndLabelNPEval')
    print('vec created')
    RNN()