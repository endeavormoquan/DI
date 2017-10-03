import gensim
import jieba
import os


class MySentences(object):
    def __init__(self,dirname):
        self.dirname = dirname

    def __iter__(self):
        for classname in os.listdir(self.dirname):
            for fname in os.listdir(os.path.join(self.dirname,classname)):
                file = open(os.path.join(self.dirname,classname,fname),'r',encoding='utf-8')
                lines = file.readlines()
                if lines.index('$\n'):
                    pos = lines.index('$\n')
                    str = lines[pos+1]
                    text = [word for word in list(jieba.cut(str))]
                    yield text


def createModel():
    sentences = MySentences('D:\QATest')
    model = gensim.models.Word2Vec(sentences,size=30,workers=1,min_count=2)
    model.save('D:\Git\model')
    # print(model.wv['牙周病'])

def createVec(dirname):
    VecTargetPath = 'D:\\VecData\\Vec.txt'
    LabelTargetPath = 'D:\\LableData\\label.txt'
    model = gensim.models.Word2Vec.load('D:\Git\model')
    fw1 = open(VecTargetPath,'a')
    fw2 = open(LabelTargetPath,'a')

    classth = 0 # used for label
    for classname in os.listdir(dirname):
        classnum = len(os.listdir(dirname))
        print(classname)
        for fname in os.listdir(os.path.join(dirname,classname)):
            file = open(os.path.join(dirname,classname,fname),'r',encoding='utf-8')
            lines = file.readlines()
            if lines.index('$\n'):
                pos = lines.index('$\n')
                str = lines[pos + 1]
                import improveBayes
                rubbishWords = improveBayes.someRubbishWords()
                text = [word for word in list(jieba.cut(str))]
                for word in text:
                    if word in rubbishWords:
                        text.remove(word)
                if len(text) >= 40:
                    Vec = []
                    Label = [0] * classnum
                    Label[classth] = 1
                    for word in text:
                        if word in model.wv:
                            Vec.append(model.wv[word])
                    from sklearn import decomposition
                    pca = decomposition.PCA(n_components=30)
                    pca.fit(Vec)
                    print(pca.components_.shape)
                    print(Label)
                    # fw1.write(pca.components_)
                    # fw2.write(Label)
        classth += 1
    fw1.close()
    fw2.close()

if __name__ == '__main__':
    # createModel()
    createVec('D:\QATest')
#todo how to batch the data(pca.components_ and Label)
