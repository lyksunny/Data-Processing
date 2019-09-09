数据处理工具记录【一】—— 数据
======
* [使用工具](#%E4%BD%BF%E7%94%A8%E5%B7%A5%E5%85%B7)
* [数据读取](#%E6%95%B0%E6%8D%AE%E8%AF%BB%E5%8F%96)
  * [读取csv文件](#%E8%AF%BB%E5%8F%96csv%E6%96%87%E4%BB%B6)
  * [读取excel文件](#%E8%AF%BB%E5%8F%96excel%E6%96%87%E4%BB%B6)
* [数据浏览](#%E6%95%B0%E6%8D%AE%E6%B5%8F%E8%A7%88)
  * [显示信息](#%E6%98%BE%E7%A4%BA%E4%BF%A1%E6%81%AF)
  * [划分测试集](#%E5%88%92%E5%88%86%E6%B5%8B%E8%AF%95%E9%9B%86)
  * [可视化](#%E5%8F%AF%E8%A7%86%E5%8C%96)
    * [频数直方图](#%E9%A2%91%E6%95%B0%E7%9B%B4%E6%96%B9%E5%9B%BE)
    * [散点图](#%E6%95%A3%E7%82%B9%E5%9B%BE)
    * [带透明度的散点图](#%E5%B8%A6%E9%80%8F%E6%98%8E%E5%BA%A6%E7%9A%84%E6%95%A3%E7%82%B9%E5%9B%BE)
    * [带热力图与大小的散点图](#%E5%B8%A6%E7%83%AD%E5%8A%9B%E5%9B%BE%E4%B8%8E%E5%A4%A7%E5%B0%8F%E7%9A%84%E6%95%A3%E7%82%B9%E5%9B%BE)
    * [保存图片](#%E4%BF%9D%E5%AD%98%E5%9B%BE%E7%89%87)
  * [寻找相关性](#%E5%AF%BB%E6%89%BE%E7%9B%B8%E5%85%B3%E6%80%A7)
    * [皮尔逊相关系数矩阵](#%E7%9A%AE%E5%B0%94%E9%80%8A%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0%E7%9F%A9%E9%98%B5)
    * [多指标散点图](#%E5%A4%9A%E6%8C%87%E6%A0%87%E6%95%A3%E7%82%B9%E5%9B%BE)
  * [数据组合](#%E6%95%B0%E6%8D%AE%E7%BB%84%E5%90%88)
* [数据清洗](#%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97)
  * [一般方法](#%E4%B8%80%E8%88%AC%E6%96%B9%E6%B3%95)
  * [特征放缩](#%E7%89%B9%E5%BE%81%E6%94%BE%E7%BC%A9)
  * [流水线](#%E6%B5%81%E6%B0%B4%E7%BA%BF)
# 使用工具
jupyter notebook
pandas、sklearn
(matplotlib和numpy、scipy就不说了)
以上都可pip安装

注： 
1. 本人使用的sklearn版本是0.18
2. jupyter notebook 安装前先升级pip，再输入pip install jupyter
3. jupyter使用时先定位到项目文件夹。使用教程可参考博客： [https://blog.csdn.net/gubenpeiyuan/article/details/79252402](https://blog.csdn.net/gubenpeiyuan/article/details/79252402)
4. 图片样例来源于：机器学习实战：基于Scikit-Learn和TensorFlow
# 数据读取
## 读取csv文件
~~~
import pandas as pd
data = pd.read_csv(path)
~~~
## 读取excel文件
~~~
# 读取第一个sheet
data = pd.read_excel(path)  

# 读取第二个sheet
data = pd.read_excel(path, sheet_name=1)  

 # 读取所有的sheet，以字典的形式保存，key为sheet名
data = pd.read_excel(path, sheet_name=-1)  
~~~
返回DataFrame类。
其他参数可以参考博客：
[https://blog.csdn.net/weixin_38546295/article/details/83537558](https://blog.csdn.net/weixin_38546295/article/details/83537558)
# 数据浏览
## 显示信息
~~~
data.head()  # 显示前5行数据
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908111036475.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
~~~
data.info()  # 显示一些信息，可以看出哪列有缺失，数据格式等
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908111134230.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
~~~
data["key"].value_counts()  # 显示该列的分类统计信息
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908111234417.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
~~~
data.describe()  # 显示统计相关信息，包含计数、均值、标准差、最大最小值、四分位数
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908111855815.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
## 划分测试集
~~~
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
~~~
## 可视化
### 频数直方图
~~~
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))  # 绘制频数直方图，bin是矩阵数
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908112623836.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
关于DataFrame.hist()更多参数说明，可参考官网：
[https://pandas.pydata.org/pandas-docs/version/0.19/generated/pandas.DataFrame.hist.html](https://blog.csdn.net/weixin_38546295/article/details/83537558)

如果打开较慢或打不开，可参考博客：
[https://blog.csdn.net/yanwucao/article/details/79841544](https://blog.csdn.net/yanwucao/article/details/79841544)
### 散点图
~~~
data.plot(kind="scatter", x=key1, y=key2)
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908113145536.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
### 带透明度的散点图
加上透明度，可以更有效的显示聚集情况
~~~
data.plot(kind="scatter", x=key1, y=key2，alpha=0.1)
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908113329426.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
### 带热力图与大小的散点图
~~~
data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", 
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908114502321.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
### 保存图片
~~~
def save_fig(path, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
~~~
## 寻找相关性
### 皮尔逊相关系数矩阵
~~~
corr_matrix = data.corr()
corr_matrix[key].sort_values(ascending=False)
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908163134209.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
### 多指标散点图
不同指标位置为对应的散点，对角线相同指标位置为频数直方图。
从这样的图里面可以看到相关性
~~~
from pandas.plotting import scatter_matrix
attributes = ["key1", "key2", "key3", "key4"]
scatter_matrix(data[attributes], figsize=(12, 8))
~~~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908114955742.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908115003195.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTkzMjA0Ng==,size_16,color_FFFFFF,t_70)
## 数据组合
~~~
data[new_key] = data[key1] / data[key2]
~~~
# 数据清洗
## 一般方法
一般有以下三种方法，再DataFrame中都有方法可调用
1. 放弃这一条数据，对应dropna()方法
2. 放弃这一条属性，对应drop()方法
3. 填补缺失值（0，中位数，平均数），对应fillna()方法
~~~
data.dropna(subset=[key1,key2])
data.drop(key, axis=1)
median = data[key].median()
data[key].fillna(median)
~~~
sklearn中有更好的估算器和转换器完成第三种方法
~~~
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
imputer.fit(data)  # 估算器，一般用来拟合,估算参数
data = imputer.transform(data)  # 转换器，根据估算器更新数据集

# 此时data的属性时numpy.ndarray数组，最好还是转换成pandas的DataFrame
data = pd.DataFrame(data)
~~~
也可以自定义转换器，需要重构fit()和transform()方法。
~~~
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names): #可以为列表
        self.attribute_names = attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values #返回的为numpy array
~~~
## 特征放缩
~~~
from sklearn.preprocessing import StandardScaler  # 标准化（减去均值再除以标准差）
from sklearn.preprocessing import MinMaxScaler  # 归一化（0~1）
~~~
## 流水线
将上述步骤合在一起，安装一定顺序执行，调用优化过的fit_transform()方法。
~~~
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([('imputer', Imputer(strategy="median")),   # 估算器
                        ('attribs_adder', CombinedAttributesAdder()),   # 转换器
                        ('std_scaler', StandardScaler())   #特征缩放
                        ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
~~~
流水线也可以组合
~~~
from sklearn.pipeline import FeatureUnion

num_attribs = list(data)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),   # 自定义的数据组合转换器
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer()),  # 文本转换器，给字符编码
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
data = full_pipeline.fit_transform(data)
~~~
