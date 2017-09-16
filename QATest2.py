import numpy as np
import jieba  # for now, i will use jieba,a third_part lib to cut the string, maybe later i will use LSTM
import os

import improveBayes

docList = []
classList = []
fullText = []
# all this three will be updated upon the loadDataSet() comes into action

def loadDataSet():
    classname = os.listdir('D:\\QATest')
    classNum = len(classname)
    for className in classname:
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
    return classNum

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

def trainNB(trainMatrix,trainCategory,numOfClasses):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pci = []
    pNum = []
    pDenom = []
    pResult = []
    for index in range(numOfClasses):
        pci.append(trainCategory.count(index)/numTrainDocs)
        pNum.append(np.ones(numWords))
        pDenom.append(2.0)
    for i in range(numTrainDocs):
        pNum[trainCategory[i]] += trainMatrix[i]
        pDenom[trainCategory[i]] += sum(trainMatrix[i])
    for index in range(numOfClasses):
        pResult.append(np.log(pNum[index]/pDenom[index]))
    return pResult,pci

def classifyNB(vec2Classify,pResult,pci):
    p = []
    for index in range(len(pci)):
        p.append(sum(np.array(vec2Classify)*np.array(pResult[index]))+np.log(pci[index]))
    temp = p[0]
    result = 0
    for i in range(len(p)):
        if p[i]>temp:
            result = i
    return result

def deseaseTest():
    classNum = loadDataSet()
    vocabList = createVocabList(docList)

    trainingSet = []
    for i in range(classNum):
        trainingSet += list(range(classList.index(i),classList.index(i)+40))
    testSet = []
    for i in range(20*classNum):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p,p_ = trainNB(trainMat,trainClasses,classNum)
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVec,p,p_) != classList[docIndex]:
            errorCount+=1
    print(errorCount)
    print(errorCount/len(testSet))

def improvedDeseaseTest():
    classNum = loadDataSet()
    vocabList = createVocabList(docList)

    freqRub = improveBayes.someRubbishWords()
    for word in freqRub:
        if word in vocabList:
            vocabList.remove(word)

    top30Wordds = improveBayes.calcMostFreq(vocabList,fullText)
    for pairW in top30Wordds:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    print(len(vocabList))

    trainingSet = []
    for i in range(classNum):
        trainingSet += list(range(classList.index(i), classList.index(i) + 40))
    testSet = []
    for i in range(20 * classNum):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(improveBayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # print(trainMat)
# here I can apply the PCA to the trainMat

    p, p_ = trainNB(trainMat, trainClasses, classNum)
    errorCount = 0
    for docIndex in testSet:
        wordVec = improveBayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(wordVec, p, p_) != classList[docIndex]:
            errorCount += 1
    print(errorCount)
    print(errorCount / len(testSet))

if __name__ == '__main__':
    improvedDeseaseTest()

