srcFilename = 'D:\\PageData.txt'
srcF = open(srcFilename,'r')

outFilename1 = 'D:\\Page3-1Data.txt'
outFilename2 = 'D:\\Page3-2Data.txt'
outFilename3 = 'D:\\Page3-3Data.txt'
outFilename4 = 'D:\\Page3-4Data.txt'

outF1 = open(outFilename1,'w')
outF2 = open(outFilename2,'w')
outF3 = open(outFilename3,'w')
outF4 = open(outFilename4,'w')
line = ''

for i in range (20020):
    outF1.write(srcF.readline())
outF1.close()

for i in range (19782):
    outF2.write(srcF.readline())
outF2.close()

for i in range (20048):
    outF3.write(srcF.readline())
outF3.close()

for i in range (16519):
    outF4.write(srcF.readline())
outF4.close()
