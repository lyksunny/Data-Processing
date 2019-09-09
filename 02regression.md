数据处理工具记录【二】—— 回归
* [选择和训练模型](#%E9%80%89%E6%8B%A9%E5%92%8C%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)
  * [线性回归模型](#%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B)
    * [普通线性回归](#%E6%99%AE%E9%80%9A%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92)
    * [多项式回归](#%E5%A4%9A%E9%A1%B9%E5%BC%8F%E5%9B%9E%E5%BD%92)
    * [岭回归](#%E5%B2%AD%E5%9B%9E%E5%BD%92)
    * [套索回归（Lasso）](#%E5%A5%97%E7%B4%A2%E5%9B%9E%E5%BD%92lasso)
    * [弹性网络](#%E5%BC%B9%E6%80%A7%E7%BD%91%E7%BB%9C)
    * [逻辑回归](#%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92)
    * [Softmax回归（多元逻辑回归）](#softmax%E5%9B%9E%E5%BD%92%E5%A4%9A%E5%85%83%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92)
    * [总结](#%E6%80%BB%E7%BB%93)
  * [SVM回归](#svm%E5%9B%9E%E5%BD%92)
  * [决策树模型](#%E5%86%B3%E7%AD%96%E6%A0%91%E6%A8%A1%E5%9E%8B)
  * [随机森林模型](#%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97%E6%A8%A1%E5%9E%8B)
  * [交叉验证](#%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81)
* [微调模型](#%E5%BE%AE%E8%B0%83%E6%A8%A1%E5%9E%8B)
  * [网络搜索](#%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2)
  * [随机搜索](#%E9%9A%8F%E6%9C%BA%E6%90%9C%E7%B4%A2)
  * [分析最佳模型及其错误](#%E5%88%86%E6%9E%90%E6%9C%80%E4%BD%B3%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E9%94%99%E8%AF%AF)
  * [通过测试集评估系统](#%E9%80%9A%E8%BF%87%E6%B5%8B%E8%AF%95%E9%9B%86%E8%AF%84%E4%BC%B0%E7%B3%BB%E7%BB%9F)
    * [早期停止法](#%E6%97%A9%E6%9C%9F%E5%81%9C%E6%AD%A2%E6%B3%95)
=====
# 选择和训练模型
## 线性回归模型
### 普通线性回归
~~~
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 计算RMSE误差（2范数）
from sklearn.metrics import mean_squared_error
y_predictions = lin_reg.predict(X)
lin_mse = mean_squared_error(y, y_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
~~~
### 多项式回归
~~~
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)  #获得X的值和X平方的值
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
~~~
### 岭回归
最小化权重L2范数
~~~
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
~~~
### 套索回归（Lasso）
最小化权重的L1范数
~~~
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
~~~
### 弹性网络
岭回归和套索回归的混合
~~~
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio对应L1范数的比例
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
~~~
### 逻辑回归
~~~
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X, y)
~~~
### Softmax回归（多元逻辑回归）
~~~
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
~~~
### 总结
1. 一般情况下岭回归比普通线性回归效果更好，而且普通线性回归求解时矩阵不一定可逆，但是岭回归的矩阵一定可逆。
2. Lasso回归倾向于将不重要的权重降至0，是一个自动执行特征选择的方法。如果不确定时少数几个特征真正重要时，应更青睐岭回归。
3. 弹性网络比Lasso更受欢迎，因为Lasso回归可能产生异常表现（比如多个特征强相关，或者特征数量比训练实例多时）。并且弹性网络可以通过调节超参数来对模型进行调整。当l1_radio接近1时，就是Lasso回归
4. 逻辑回归被广泛应用于估算一个实例属于某个特定的概率。
5. Softmax回归的成本函数不是预测与实际的欧氏距离，而是交叉熵。
## SVM回归
~~~
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)

# 加核函数
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)
~~~
## 决策树模型
~~~
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
~~~

## 随机森林模型
~~~
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
~~~
## 交叉验证
交叉验证将数据集分为cv个不同的子集，然后对模型进行cv次训练和评估——每次选择一个子集评估，其他的用来训练。
~~~
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
~~~
Scikit-Learn的交叉验证功能更倾向于使用越大越好的效用函数，而不是越小越好的成本函数，这也是为什么计算出来的scores是负的。
# 微调模型
## 网络搜索
需要知道该模型的超参数
~~~
# 随机森林调参
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)

#以下三句在三个块里分别执行
# 显示最佳参数组合
grid_search.best_params_
# 显示最佳模型
grid_search.best_estimator_
#评估分数
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]): 
	print(np.sqrt(-mean_score), params)
~~~
## 随机搜索
RandomizedSearchCV()方法，于网格搜索大致相同
## 分析最佳模型及其错误
~~~
feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, attributes), reverse=True)
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908174743670.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
有了这些信息就可以尝试删除不太有用的特征。
## 通过测试集评估系统
~~~
final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test， y_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
~~~
### 早期停止法
防止对训练集过拟合而测试集误差偏大
~~~
n_epochs = 500
train_errors, val_errors = [], []
for epoch in range(n_epochs):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))

best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])
~~~
