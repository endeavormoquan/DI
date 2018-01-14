def test_Lasso_alpha(*data):
    X_train,X_test,y_train,y_test=data
    alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores = []
    for i,alpha in enumerate(alphas):
        regr = linear_model.Lasso(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test,y_test))
    ##绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas,socres)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_xscale('log')
    ax.set_title("Lasso")
    plt.show()

X_train, X_test, y_train, y_test = load_data()
test_Lasso_alpha(X_train, X_test, y_train, y_test)