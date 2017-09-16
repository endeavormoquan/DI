import numpy as np
import jieba
import os

docList = []
classList = []
fullText = []

def loadDataSet():
    classNum = os.listdir('D:\\QATest')
    for className in classNum:
        fileList = os.listdir('D:\\QATest\\'+className)
        for filename in fileList:
            fileRoute = 'D:\\QATest\\'+className+'\\'+filename
            file = open(fileRoute,'r',encoding='utf-8')
            lines = file.readlines()
            if lines.index('$\n'):
                pos = lines.index('$\n')
                str = lines[pos+1]
                wordList = list(jieba.cut(str))
                docList.append(wordList)
                fullText.extend(wordList)
                classList.append(int(className))
            file.close()

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet|set(document)
    return list(vocabSet)

def setOfWord2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('not found')
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    p_ = sum(trainCategory)/numTrainDocs
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,p_

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(np.array(vec2Classify)*np.array(p1Vec))+np.log(pClass1)
    p0 = sum(np.array(vec2Classify)*np.array(p0Vec))+np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def deseaseTest():
    loadDataSet()
    vocabList = createVocabList(docList)
    trainingSet = list(range(40,80)) + list(range(350,410))
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,p_ = trainNB0(trainMat,trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVec,p0v,p1v,p_) != classList[docIndex]:
            errorCount+=1
    print(errorCount)
    print(errorCount/len(testSet))

if __name__ == '__main__':
    deseaseTest()




