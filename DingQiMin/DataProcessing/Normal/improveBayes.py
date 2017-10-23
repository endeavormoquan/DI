import numpy as np
import os

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

# todo add the PCA or SVD to improve the accuracy
def pca(dataMat,topNfeat = 9999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = np.cov(meanRemoved, rowvar=False)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 得到两个值，一个是特征值，一个是特征矩阵。。我们要拿到特征矩阵。
    # print(eigVals)
    eigValInd = np.argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 这个是在重构矩阵，用我们的低纬度矩阵。重构矩阵还是有差别的。
    return lowDDataMat, reconMat

def someRubbishWords():
    fr = open('D:\Git\DI\\DingQiMin\\freqRubbishword.txt')
    words = set([])
    lines = fr.readlines()
    for line in lines:
        words = words|set(line.strip())
    words = words|set('\n')
    return list(words)

def deleteRubbishWords(vocabList):
    freqRub = someRubbishWords()
    for word in freqRub:
        if word in vocabList:
            vocabList.remove(word)
    return vocabList

def improvedCreateVocabList(docList):
    vocabSet = set([])
    for document in docList:
        vocabSet = vocabSet | set(document)
    vocabList = list(vocabSet)
    vocabList = deleteRubbishWords(vocabList)
    return vocabList

if __name__ == '__main__':
    someRubbishWords()