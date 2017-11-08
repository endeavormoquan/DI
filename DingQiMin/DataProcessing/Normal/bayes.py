import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWord2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word is not in the vocabList")
    print(returnVec)
    return returnVec

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / numTrainDocs
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

    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    print(p1Vect)
    print(p0Vect)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(np.array(vec2Classify) * np.array(p1Vec)) + np.log(pClass1)
    p0 = sum(np.array(vec2Classify) * np.array(p0Vec)) + np.log(1-pClass1)
    print(p1)
    print(p0)
    if p1>p0:
        return 1
    else:
        return 0

def testNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList,postinDoc))
    p0v,p1v,pAb = trainNB0(trainMat,listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWord2Vec(myVocabList,testEntry)
    print(classifyNB(thisDoc,p0v,p1v,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = setOfWord2Vec(myVocabList, testEntry)
    print(classifyNB(thisDoc, p0v, p1v, pAb))

def testParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    fileRoute = 'D:\MachineLearininginAction\Ch04\\'
    for i in range(1,26):
        filename = fileRoute+'email\spam\%d.txt'%i
        wordList = testParse(open(filename,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        filename = fileRoute + 'email\ham\%d.txt' % i
        wordList = testParse(open(filename, 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam = trainNB0(trainMat,trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVec,p0v,p1v,pSpam) != classList[docIndex]:
            errorCount += 1
    print(errorCount/len(testSet))

# if __name__ == '__main__':
#     testNB()