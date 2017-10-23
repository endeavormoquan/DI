from urllib import request
from urllib import error
from http import cookiejar
import re
import time
import os
import numpy as np
import requests


def get_ip():
    ipFile = open('D:\\IpList.txt','r')
    ipList = []
    ip = '!'
    while ip != '':
        ip = ipFile.readline()
        ipList.append(ip)
    ipList = ipList[:-1]
    IP = ''.join(np.random.choice(ipList)).strip()
    return(IP)

headerList = ["Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
 "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
 "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
 "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
 "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
 "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
 "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
 "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
 "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
 "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
 "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
 "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
 "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
 "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
 "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
 "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
 "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
 "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"]
cookies = cookiejar.CookieJar()
opener = request.build_opener(request.HTTPCookieProcessor(cookies))

srcFilename = 'D:\\Page3-1Data.txt'
srcF = open(srcFilename,'r')
Length = len(srcF.readlines())
srcF.seek(0)

SFname = 'D:\\QAPairData\\sourceFile1.txt'

fileCount = 0
readStr = srcF.readline()[:-1]

failCount = 0
while readStr != '':

    if readStr == '!':
        DiseaseName = srcF.readline()[:-1] #  疾病名称
        print(DiseaseName)
        fileCount = 0
    else: # 不可能再有！和疾病名称
        print('尝试打开' + readStr)
        headers = np.random.choice(headerList)
        headers = {"User-Agent": headers}
        # ipUsed = get_ip()
        # proxy = {'http':'http://'+ipUsed}
        # proxy_handler = request.ProxyHandler(proxy)
        # opener = request.build_opener(proxy_handler)
        try:
            failCount = failCount+1
            myRequest = request.Request(url=readStr,headers=headers)
            result = opener.open(myRequest)
        except TimeoutError as e:
            print('打开页面超时')
        except error.HTTPError as e:
            print('页面打开失败')
            print(e.reason)
        except ConnectionResetError as e:
            print('页面打开失败')
        except error.URLError as e:
            print('页面打开失败')
            print(e.reason)
        else:
            failCount = 0
            try:
                hhhh = str(result.read().decode('utf-8', 'replace'))
                SF = open(SFname, 'w', encoding='utf-8')
                SF.write(hhhh)
                SF.close()
                SF = open(SFname,'r',encoding='utf-8')
                sourceResult = ''
                for line in SF:
                    sourceResult = sourceResult+line.strip()
                SF.close()
                SF = open(SFname, 'w', encoding='utf-8')
                SF.write(sourceResult)
                SF.close()
            except UnicodeDecodeError as e:
                print('页面解码失败')
                print(e.reason)
            else:
                fileCount = fileCount + 1
                print(fileCount)
                outFilename = 'D:\\QAPairData\\' + DiseaseName + '-' + str(fileCount) + '.txt'
                outF = open(outFilename, 'w', encoding='utf-8')

                patternForAskCont = re.compile(r'<p class="ask_tit">(.*?)</p>',re.S)  # only one result a page
                patternForGender = re.compile(r'<span>([男女])</span>',re.S)  # only one result a page
                patternForAge = re.compile(r'<span>(\d+)岁',re.S)  # only one result a page
                patternForAskMs = re.compile(r'<p class="txt_ms">(.*?)</p>',re.S)  # only one result a page
                patternForTags = re.compile(r'<span ><a href="/browse/.*?.html">(.*?)</a></span>',re.S)  # not only one result a page
                patternForAnswers = re.compile(r'<p class="sele_txt">(.*?)</p>',re.S)  # not only one result a page

                try:
                    resultForAskCont = re.findall(patternForAskCont,sourceResult)[0]  # only one result a page
                    outF.write('!\n')
                    outF.write(resultForAskCont.strip() + '\n')
                except IndexError as e:
                    print('resultForAskCont')
                    print(e)

                try:
                    resultForGender = re.findall(patternForGender, sourceResult)[0] # only one result a page
                    outF.write('@\n')
                    outF.write(resultForGender.strip() + '\n')
                except IndexError as e:
                    print('resultForGender')
                    print(e)

                try:
                    resultForAge = re.findall(patternForAge, sourceResult)[0]  # only one result a page
                    outF.write('#\n')
                    outF.write(resultForAge.strip() + '\n')
                except IndexError as e:
                    print('resultForAge')
                    print(e)

                try:
                    resultForAskMs = re.findall(patternForAskMs, sourceResult)[0]  # only one result a page
                    Str = resultForAskMs.strip()
                    Str = Str.replace(' ','')
                    Str = Str.replace('\t','')
                    Str = Str.replace('\v','')
                    Str = Str.replace('<br />', '')
                    Str = Str.replace('<br/>','')
                    Str = Str[:Str.find('<')]
                    outF.write('$\n')
                    outF.write(Str + '\n')
                except IndexError as e:
                    print('resultForAskMs')
                    print(e)

                resultForTags = re.findall(patternForTags,sourceResult)  # not only one result a page
                resultForAnswers = re.findall(patternForAnswers,sourceResult)  # not only one result a page

                for item in resultForTags:
                    outF.write('%\n')
                    outF.write(item.strip()+'\n')
                for item in resultForAnswers:
                    Str = item.strip()
                    Str = Str.replace('\t', '')
                    Str = Str.replace('\v', '')
                    Str = Str.replace('<br />','')
                    Str = Str.replace('<br/>', '')
                    Str = Str[:Str.find('<')]
                    outF.write('&\n')
                    outF.write(Str+'\n')
                outF.close()

    readStr = srcF.readline()[:-1]
    time.sleep(3)
    if failCount == 10:
        print('sleep')
        time.sleep(7200)
os.remove('D:\\QAPairData\\sourceFile1.txt')



