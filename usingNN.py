import numpy as np
import tensorflow as tf
import os
import jieba.posseg as pseg

import improveBayes

docList = []
classList = []
fullText = []
classNum = 0
lengthOfEachDoc = 0

def loadDataSet():
    classname = os.listdir('D:\\QATest')
    global classNum
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
    for doc in docList:
        doc = improveBayes.bagOfWords2VecMN(vocabList,doc)
    global lengthOfEachDoc
    lengthOfEachDoc = len(doc)

if __name__ == '__main__':
    loadDataSet()



