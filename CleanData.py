import os


def cleanData():
    sourceFilename = 'D:\\QAPairData'
    fileList = os.listdir(sourceFilename)
    filesToBeRemoved = []
    for file in fileList:
        filename = 'D:\\QAPairData\\'+file
        fileOpened = open(filename,'r',encoding='utf-8')
        line = fileOpened.readline()
        answerList = []
        resultForAskCont=''
        resultForAskMs=''
        panju2 = 0
        while line != '' :
            if line == '!\n':
                resultForAskCont = fileOpened.readline()
            if line == '$\n':
                resultForAskMs = fileOpened.readline()
            if line == '&\n':
                resultForAnswers = fileOpened.readline()
                answerList.append(resultForAnswers)
            line = fileOpened.readline()
        panju1 = len(resultForAskCont+resultForAskMs)
        for item in answerList:
            panju2 = panju2 + len(item)
        if panju1 < 10:
            filesToBeRemoved.append(filename)
            # print('delete '+filename)
        elif panju2 < 10:
            filesToBeRemoved.append(filename)
            # print('delete ' + filename)
        fileOpened.close()
    # for item in filesToBeRemoved:
    #     print(item)
    print(len(filesToBeRemoved))
    for item in filesToBeRemoved:
        os.remove(item)

if __name__ == '__main__':
    cleanData()