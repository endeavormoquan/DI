#Lasso_alpha测试函数
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,discriminant_analysis ,model_selection

#模型


def load_data():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data,diabetes.target,test_size=0.25,random_state=0)

def test_Lasso_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores=[]
    for i,alpha in enumerate(alphas):
        regr = linear_model.Lasso(alpha=alpha)   #调用Lasso类中linear_model.Lasso函数
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))

    #绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Lasso")
    plt.show()

X_train,X_test,y_train,y_test=load_data()
test_Lasso_alpha(X_train,X_test,y_train,y_test)