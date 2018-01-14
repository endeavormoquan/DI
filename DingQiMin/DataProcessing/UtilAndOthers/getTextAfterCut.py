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
            for fname in os.listdir(os.path.join(self.dirname,classname)):
                print(fname)
                file = open(os.path.join(self.dirname,classname,fname),'r',encoding='utf-8')
                lines = file.readlines()
                if lines.index('$\n'):
                    pos = lines.index('$\n')
                    str = lines[pos+1]
                    text = [word for word in list(jieba.cut(str))] # 此处的分词算法可以替换成自己的
                    yield text


def afterCut(srcDir,disDir):
    classth = 0  # used for label
    for classname in os.listdir(srcDir):
        classnum = len(os.listdir(srcDir))
        fileth = 0
        for fname in os.listdir(os.path.join(srcDir, classname)):
            file = open(os.path.join(srcDir, classname, fname), 'r', encoding='utf-8')
            lines = file.readlines()
            if lines.index('$\n'):
                pos = lines.index('$\n')
                str = lines[pos + 1]

                rubbishWords = improveBayes.someRubbishWords()
                text = [word for word in list(jieba.cut(str))]
                for word in text:
                    if word in rubbishWords:
                        text.remove(word)

        classth += 1