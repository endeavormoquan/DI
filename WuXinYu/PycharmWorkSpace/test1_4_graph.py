#ElasticNet_alpha测试函数
import matplotlib.pyplot as plt
import matplotlib as npl
import numpy as np
from sklearn import datasets, linear_model,discriminant_analysis ,model_selection

def load_data():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data,diabetes.target,test_size=0.25,random_state=0)

def test_ElasticNet_alpha_rho(*data):
    X_train, X_test, y_train, y_test = data
    alphas=np.logspace(-2,2)
    rhos=np.linspace(0.01,1)
    scores=[]
    for alpha in alphas:
        for rho in rhos:
            regr = linear_model.ElasticNet(alpha=alpha, l1_ratio=rho)   #alpha和l1_ratio是模型中对应两个参数的数值
            regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))

    #绘图
    alphas, rhos =np.meshgrid(alphas, rhos)
    scores=np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import pyplot as plt
    fig=plt.figure()
    ax=Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=Flase)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel('score')
    ax.set_title("ElasticNet")
    plt.show()

X_train,X_test,y_train,y_test=load_data()
test_ElasticNet_alpha_rho(X_train,X_test,y_train,y_test)