数据处理工具记录【三】—— 分类
======
* [选择和训练模型](#%E9%80%89%E6%8B%A9%E5%92%8C%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)
  * [SGD模型](#sgd%E6%A8%A1%E5%9E%8B)
  * [SVM模型（支持向量机）](#svm%E6%A8%A1%E5%9E%8B%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)
    * [线性SVM](#%E7%BA%BF%E6%80%A7svm)
    * [非线性SVM](#%E9%9D%9E%E7%BA%BF%E6%80%A7svm)
      * [多项式核](#%E5%A4%9A%E9%A1%B9%E5%BC%8F%E6%A0%B8)
      * [高斯RBF核](#%E9%AB%98%E6%96%AFrbf%E6%A0%B8)
  * [决策树模型](#%E5%86%B3%E7%AD%96%E6%A0%91%E6%A8%A1%E5%9E%8B)
  * [随机森林模型](#%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97%E6%A8%A1%E5%9E%8B)
  * [多类别分类器](#%E5%A4%9A%E7%B1%BB%E5%88%AB%E5%88%86%E7%B1%BB%E5%99%A8)
    * [强制使用OvO](#%E5%BC%BA%E5%88%B6%E4%BD%BF%E7%94%A8ovo)
  * [多标签分类](#%E5%A4%9A%E6%A0%87%E7%AD%BE%E5%88%86%E7%B1%BB)
    * [KNN模型](#knn%E6%A8%A1%E5%9E%8B)
  * [多输出分类](#%E5%A4%9A%E8%BE%93%E5%87%BA%E5%88%86%E7%B1%BB)
* [性能考核](#%E6%80%A7%E8%83%BD%E8%80%83%E6%A0%B8)
  * [交叉验证](#%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81)
  * [混淆矩阵](#%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5)
  * [精度和召回率](#%E7%B2%BE%E5%BA%A6%E5%92%8C%E5%8F%AC%E5%9B%9E%E7%8E%87)
  * [决策阈值](#%E5%86%B3%E7%AD%96%E9%98%88%E5%80%BC)
  * [ROC曲线](#roc%E6%9B%B2%E7%BA%BF)
  * [ROC AUC](#roc-auc)
# 选择和训练模型
## SGD模型
~~~
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])
~~~
## SVM模型（支持向量机）
### 线性SVM
~~~
from sklearn.svm import LinearSVC
svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42))
))

svm_clf.fit(X, y)
~~~
### 非线性SVM
#### 多项式核
~~~
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
))
poly_kernel_svm_clf.fit(X, y)
~~~
#### 高斯RBF核
添加相似特征的思想
~~~
rbf_kernel_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
))
rbf_kernel_svm_clf.fit(X, y)
~~~
## 决策树模型
~~~
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
~~~
可视化
~~~
from sklearn.tree import export_graphviz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

export_graphviz(
    tree_clf,
    out_file=image_path("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
~~~
然后使用命令行
~~~
dot -Tpng iris_tree.dot -o iris_tree.png
~~~
## 随机森林模型
~~~
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
~~~
返回一个数组，每行为一个实例，每列代表一个类别。意思是某个给定实例属于某个给定类别的概率。
## 多类别分类器
1. OvA 一个实例给n个类别的判定，判定分数高的归为该类
2. OvO 类别两两判定，n个类别会产生(n-1)*n/2个分类器
Scikit-Learn会自动执行OvA，SVM除外
### 强制使用OvO
以SGD为例
~~~
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
~~~
len(ovo_clf.estimators_) = 45
注：随机森林直接可以分为多个类别
## 多标签分类
### KNN模型
~~~
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])  # 会花很长时间，可能有几个小时

# 评估：每个标签下的发f1分数的平均值
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
~~~
## 多输出分类
KNN模型可以实现
# 性能考核
## 交叉验证
~~~
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
~~~
交叉验证在分类模型中不是很好的判断方式。
## 混淆矩阵
~~~
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
~~~
## 精度和召回率
~~~
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

# 精度和召回率的调和平均值： f1分数
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
~~~
## 决策阈值
~~~
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
~~~
## ROC曲线
~~~
from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
plot_roc_curve(fpr, tpr)
plt.show()
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908213253381.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
## ROC AUC
ROC曲线下面面积
~~~
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
~~~
