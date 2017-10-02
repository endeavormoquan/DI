import gensim
import jieba
def Topics_and_Transformations():
    documents = ['我是丁启民','我来自西北工业大学','我今年21岁了']
    texts = [[word for word in list(jieba.cut(document))] for document in documents]
    model = gensim.models.Word2Vec(texts, size=10, workers=1, min_count=1)
    model.save('D:\Git\DI\model')
    print(texts)
    for doc in texts:
        for word in doc:
            print(model.wv[word])

if __name__ == '__main__':
    Topics_and_Transformations()