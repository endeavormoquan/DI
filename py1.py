import os


sourceFilename = 'D:\\QAPairData'
fileList = os.listdir(sourceFilename)
patientFile = 'D:\\QAPairData\\ApatientFile.txt'
patientF = open(patientFile,'w',encoding='utf-8')
doctorFile = 'D:\\QAPairData\\AdoctorFile.txt'
doctorF = open(doctorFile,'w',encoding='utf-8')
for filename in fileList:
    print(filename)
    file = open('D:\\QAPairData\\'+filename,'r',encoding='utf-8')
    line = file.readline()
    while line != '':
        # if line == '!\n':
        #     resultForAskCont = file.readline()
        #     patientF.write(resultForAskCont)
        if line == '$\n':
            resultForAskMs = file.readline()
            patientF.write(resultForAskMs)
        if line == '&\n':
            resultForAnswers = file.readline()
            doctorF.write(resultForAnswers)
        line = file.readline()
    file.close()
patientF.close()
doctorF.close()
