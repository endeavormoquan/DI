import gensim
import jieba
import os


class MySentences(object):
    def __init__(self,dirname):
        self.dirname = dirname

    def __iter__(self):
        for classname in os.listdir(self.dirname):
            for fname in os.listdir(os.path.join(self.dirname,classname)):
                # print(fname)
                file = open(os.path.join(self.dirname,classname,fname),'r',encoding='utf-8')
                lines = file.readlines()
                if lines.index('$\n'):
                    pos = lines.index('$\n')
                    str = lines[pos+1]
                    text = [word for word in list(jieba.cut(str))]
                    yield text

def func():
    sentences = MySentences('D:\QATest')
    for index in sentences:
        print(index)
    # model = gensim.models.Word2Vec(sentences)

if __name__ == '__main__':
    func()
