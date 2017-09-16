import numpy as np

def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = []
    # stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # print(stringArr)
    # datArr = [map(float, line) for line in stringArr]
    # return np.mat(datArr)
    lines = fr.readlines()
    for line in lines:
        tempList = line.strip().split('\t')
        for index in range(len(tempList)):
            tempList[index] = float(tempList[index])
        stringArr.append(tempList)
    stringArr = np.array(stringArr)
    return stringArr

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


if __name__ == '__main__':
    dataMat = loadDataSet('D:\MachineLearininginAction\Ch13\\testSet.txt')
    lowDMat,reconMat = pca(dataMat,1)