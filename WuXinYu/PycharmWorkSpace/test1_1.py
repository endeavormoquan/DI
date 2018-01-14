#线性回归
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,discriminant_analysis,model_selection

#模型
#class sklearn.linear_model.LinearRefression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

def load_data():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data,diabetes.target,test_size=0.25,random_state=0)


def test_LinearRegression(*data):
    X_train,X_test,y_train,y_test=data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
    print("Residual sum of squares:%.2f"% np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' %regr.score(X_test, y_test))


X_train,X_test,y_train,y_test=load_data()
test_LinearRegression(X_train,X_test,y_train,y_test)

