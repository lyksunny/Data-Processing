数据处理工具记录【四】—— 降维
======
* [PCA](#pca)
  * [简单实现](#%E7%AE%80%E5%8D%95%E5%AE%9E%E7%8E%B0)
    * [奇异值分解](#%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3)
    * [低维度投影](#%E4%BD%8E%E7%BB%B4%E5%BA%A6%E6%8A%95%E5%BD%B1)
    * [sklearn](#sklearn)
    * [方差解释率](#%E6%96%B9%E5%B7%AE%E8%A7%A3%E9%87%8A%E7%8E%87)
  * [选择正确数量的维度](#%E9%80%89%E6%8B%A9%E6%AD%A3%E7%A1%AE%E6%95%B0%E9%87%8F%E7%9A%84%E7%BB%B4%E5%BA%A6)
  * [调用外存](#%E8%B0%83%E7%94%A8%E5%A4%96%E5%AD%98)
    * [增量PCA(IPCA)](#%E5%A2%9E%E9%87%8Fpcaipca)
    * [随机PCA](#%E9%9A%8F%E6%9C%BApca)
  * [核主成分分析](#%E6%A0%B8%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)
    * [使用RBF核函数](#%E4%BD%BF%E7%94%A8rbf%E6%A0%B8%E5%87%BD%E6%95%B0)
    * [选择核函数和调整超参数](#%E9%80%89%E6%8B%A9%E6%A0%B8%E5%87%BD%E6%95%B0%E5%92%8C%E8%B0%83%E6%95%B4%E8%B6%85%E5%8F%82%E6%95%B0)
* [局部线性嵌入](#%E5%B1%80%E9%83%A8%E7%BA%BF%E6%80%A7%E5%B5%8C%E5%85%A5)
# PCA
## 简单实现
### 奇异值分解
~~~
import numpy as np
X_centered = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X_centered)
c1 = V.T[:, 0]  # 第一个成分
c2 = V.T[:, 1]  # 第二个成分
~~~
### 低维度投影
~~~
W2 = V.T[:, :2]
X2D = X_centered.dot(W2)
~~~
### sklearn
~~~
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
~~~
### 方差解释率
~~~
print(pca.explained_variance_ratio_)
~~~
## 选择正确数量的维度
~~~
pca = PCA(n_compoents=0.95)
X_reduced = pca.fit_transform(X)
~~~
## 调用外存
### 增量PCA(IPCA)
~~~
from sklearn.decomposition import IncermentalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_compoents=154)
for X_batch in np.array_split(X_mnist, n_batch):
	inc_pca.partial_fit(X_batch)
X_mnist_reduced = inc_pca.transfrom(X_mnist)

# 或者使用numpy里的memmap类
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m,n))
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_componemts=154, batch_size=batch_size)
inc_pca.fit(X_mm)
~~~
### 随机PCA
~~~
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd.pca.fit_transform(X_mnist)
~~~
## 核主成分分析
### 使用RBF核函数
~~~
from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
~~~
### 选择核函数和调整超参数
~~~
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
	("kpca", KernelPCA(n_components=2)),
	("log_reg", LogisticRegression())
])
param_grid = [{
	"kpca_gamma": np.linspace(0.03, 0.05, 10),
	"kpca_kernel": ["rbf", "sigmoid"]
}]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X,y)
print(grid_search.best_params_)
~~~
# 局部线性嵌入
~~~
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
~~~
