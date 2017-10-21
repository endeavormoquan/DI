import os

import jieba.posseg as pseg
import numpy as np
import tensorflow as tf

from DingQiMin import improveBayes


def loadDataSet():
    docList = []
    classList = []
    fullText = []
    # classNum = 0
    # lengthOfEachDoc = 0

    classname = os.listdir('D:\\QATest')
    classNum = len(classname)
    for className in classname:
        fileList = os.listdir('D:\\QATest\\'+className)
        for filename in fileList:
            fileRoute = 'D:\\QATest\\' + className + '\\' + filename
            file = open(fileRoute,'r',encoding='utf-8')
            lines = file.readlines()
            wordList = []
            if lines.index('$\n'):
                pos = lines.index('$\n')
                str = lines[pos+1]
                words = pseg.cut(str)
                for w in words:
                    if w.flag == 'v' or w.flag == 'n':
                        wordList.append(w.word)
                if len(wordList) < 5:
                    continue
                docList.append(wordList)
                fullText.extend(wordList)
                classList.append(int(className))
            file.close()
    vocabList = improveBayes.improvedCreateVocabList(docList)

    for index in range(len(docList)):
        vec = improveBayes.bagOfWords2VecMN(vocabList, docList[index])
        docList[index] = vec
    print(len(docList))
    print(len(docList[0]))
    lengthOfEachDoc = len(docList[0])
    return docList,classList,fullText,classNum,lengthOfEachDoc

def usingWxb(docList,classList,lengthOfEachDoc):
    batch_size = 20
    dataset_size = len(docList)
    result = []
    # entropy = []

    w1 = tf.Variable(tf.random_normal([lengthOfEachDoc,(lengthOfEachDoc//1000+1)*1000],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([(lengthOfEachDoc//1000+1)*1000,1],stddev=1,seed=1))

    x = tf.placeholder(tf.float32,shape=(None,lengthOfEachDoc),name = 'x_input')
    y_ = tf.placeholder(tf.float32,shape=(None,1),name='y_input')

    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)
#todo softmax y
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.reduce_sum((y_ - y)**2)
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    #start the session
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        STEPS = 1500
        for i in range(STEPS):
            start = (i * batch_size) % dataset_size
            end = min(start+batch_size,dataset_size)
            sess.run(train_step,feed_dict={x:docList[start:end],y_:classList[start:end]})
            if i%100 == 0:
                total_cross_entropy = sess.run(cross_entropy,feed_dict={x:docList,y_:classList})
                print(i,':',total_cross_entropy)
                # print(sess.run(y,feed_dict={x:docList[start:end],y_:classList[start:end]}))
                # entropy.append(total_cross_entropy)
                result.append(sess.run([w1,w2]))
    return result

def prepareForNN(docList,classList):
    arr = np.random.permutation(len(docList))
    docListNew = []
    classListNew = []
    for index in range(len(docList)):
        docListNew.append(docList[arr[index]])
        tempList = []
        tempList.append(classList[arr[index]])
        classListNew.append(tempList)
    return docListNew,classListNew

def testWxb(coff,docList,classList,lengthOfEachDoc):
    errorCount = 0
    w1 = np.array(coff[0])
    w2 = np.array(coff[1])
    for index in list(np.random.permutation(len(docList))[:100]):
        temp = np.array(docList[index]).reshape(1,lengthOfEachDoc)
        predict = np.dot(temp,w1)
        predict = np.dot(predict,w2)
        if predict[0][0] < 0.5:
            result = 0
        else:
            result = 1
        print(predict[0][0],result)
        if result != classList[index]:
            errorCount += 1
    print(errorCount/len(docList))


if __name__ == '__main__':
    docList, classList, fullText, classNum, lengthOfEachDoc = loadDataSet()
    docList,classList = prepareForNN(docList,classList)
    # print(np.shape(np.array(docList[0])))
    result = usingWxb(docList[:300],classList[:300],lengthOfEachDoc)
    testWxb(result[-1],docList[300:],classList[300:],lengthOfEachDoc)



