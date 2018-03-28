import gensim
import jieba
import numpy as np
import tensorflow as tf


def someRubbishWords(dir):
    fr = open(dir)
    words = set([])
    lines = fr.readlines()
    for line in lines:
        words = words|set(line.strip())
    words = words|set('\n')
    return list(words)


def func(string,gensimModel,rubbishWordsDir,cnnModel):
    model = gensim.models.Word2Vec.load(gensimModel)
    str = string
    text = [word for word in list(jieba.cut(str))]
    rubbishWords = someRubbishWords(rubbishWordsDir)
    for word in text:
        if word in rubbishWords:
            text.remove(word)
    if len(text) < 40:
        result = "字符长度不够，请重新输入"
    if len(text) >= 40:
        tempVec = []
        for word in text:
            if word in model.wv:
                tempVec.append(model.wv[word])
        from sklearn import decomposition
        pca = decomposition.PCA(n_components=28)
        pca.fit(tempVec)
        vec = pca.components_
        try:
            vec = vec.reshape([28 * 28])
            data = []
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(cnnModel)
                saver.restore(sess, tf.train.latest_checkpoint('D:/Departments/'))
                graph = tf.get_default_graph()
                x = graph.get_tensor_by_name("x:0")
                feed_dict = {x: data}
                logits = graph.get_tensor_by_name("logits_eval:0")
                classification_result = sess.run(logits, feed_dict)
                result = sess.run(tf.argmax(classification_result, 1))
                print(type(result))
        except ValueError:
            result = "字符长度不够，请重新输入"



if __name__ == '__main__':
    string = '我是丁启民，我来自西北工业大学，我今年大三了，21岁的我充满活力！我是丁启民，我来自西北工业大学，我今年大三了，21岁的我充满活力！我是丁启民，我来自西北工业大学，我今年大三了，21岁的我充满活力！我是丁启民，我来自西北工业大学，我今年大三了，21岁的我充满活力！'
    func(string,'D:\Departments\model','D:\Git\DI\DingQiMin\DataProcessing\\Normal\\freqRubbishword.txt','D:\Departments\cnnModel.ckpt.meta')