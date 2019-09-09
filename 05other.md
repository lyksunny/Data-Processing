数据处理工具记录【五】
=======
* [拟合指定曲线](#%E6%8B%9F%E5%90%88%E6%8C%87%E5%AE%9A%E6%9B%B2%E7%BA%BF)
* [解常微分方程](#%E8%A7%A3%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B)
  * [数值解](#%E6%95%B0%E5%80%BC%E8%A7%A3)
  * [符号解](#%E7%AC%A6%E5%8F%B7%E8%A7%A3)
* [拟合微分方程](#%E6%8B%9F%E5%90%88%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B)
* [假设检验](#%E5%81%87%E8%AE%BE%E6%A3%80%E9%AA%8C)
  * [T检验](#t%E6%A3%80%E9%AA%8C)
    * [单样本T检验(ttest\_1samp)](#%E5%8D%95%E6%A0%B7%E6%9C%ACt%E6%A3%80%E9%AA%8Cttest_1samp)
    * [两独立样本T检验(ttest\_ind)](#%E4%B8%A4%E7%8B%AC%E7%AB%8B%E6%A0%B7%E6%9C%ACt%E6%A3%80%E9%AA%8Cttest_ind)
    * [配对样本T检验(ttest\_rel)](#%E9%85%8D%E5%AF%B9%E6%A0%B7%E6%9C%ACt%E6%A3%80%E9%AA%8Cttest_rel)
  * [F检验](#f%E6%A3%80%E9%AA%8C)
# 拟合指定曲线
~~~
import numpy as np
from scipy.optimize import curve_fit
def func(X, paras):
	a,b = paras
	x1, x2 = X
	return a*np.exp(x1+b/x2)
popt, pcov = curve_fit(func,x,y)  # 返回参数和协方差矩阵
~~~
# 解常微分方程
## 数值解
~~~
from scipy.integrate import odeint

def dmove(compartment, t, parameters):
    alpha, beta = parameters
    S, I = compartment
    return np.array([-beta*S*I,(beta*S-alpha)*I])

N = 1339724852
S0 = N-1e-6
I0 = N-S0+1e-6
alpha, beta = 1/7, 4.6313e-11
t = np.arange(len(data_handle["全国每日报告病例数"]))
ans = odeint(dmove, (S0, I0), t, args=([alpha, beta],))  # 返回S,I的数值解
~~~
## 符号解
~~~
import sympy as sy

def differential_equation(x,f):
    return sy.diff(f(x),x,2)+f(x)#f(x)''+f(x)=0 二阶常系数齐次微分方程
x=sy.symbols('x')#约定变量
f=sy.Function('f')#约定函数
print(sy.dsolve(differential_equation(x,f),f(x)))#打印
sy.pprint(sy.dsolve(differential_equation(x,f),f(x)))#漂亮的打印
~~~
输出：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190909221508194.png)
# 拟合微分方程
~~~
from scipy.optimize.optimize import fmin
from scipy.optimize.minpack import leastsq

class Opt_ODE_para:
    def func(self, output_y, input_x, para):
        # write your ODE function in here
        a1, a2 = para
        dy1 = output_y[1]
        dy2 = a1*output_y[0] + a2*output_y[1]
        return np.array([dy1, dy2])
    def target_func_for_leastsq(self, para, output_y, input_x):
        expextedy = solve_ODE_func(output_y[0,:], input_x, para, output_y)
        error = (sum((output_y - expextedy)**2)/len(output_y))
        return error
    def target_func_for_fmin(self, para, output_y, input_x):
        expextedy = solve_ODE_func(output_y[0,:], input_x, para, output_y)
        error = (sum((output_y - expextedy)**2)/len(output_y))
        return error
    def solve_func_for_leatsq(self, yinit, t, para, y):
        return odeint(func, yinit, t, para)
    def Opt_para_using_leastsq(self, target_func_for_leastsq, para, y,x):
        return leastsq(target_func_for_leastsq, para, arg=(y,x))
    def Opt_para_using_fmin(self, target_func_for_fmin, para, y,x):
        return leastsq(target_func_for_fmin, para, arg=(y,x))      
~~~
使用时继承该类，重构func函数，调用Opt_para_using_fmin()执行。
原文档：[https://wenku.baidu.com/view/7ce29982b7360b4c2e3f64b1.html](https://wenku.baidu.com/view/7ce29982b7360b4c2e3f64b1.html)
# 假设检验
## T检验
返回t值和p值
### 单样本T检验(ttest_1samp)
~~~
from scipy import stats
stats.ttest_1samp(data,1)
~~~
### 两独立样本T检验(ttest_ind)
使用ttest_ind()函数可以进行两独立样本T检验。
当两总体方差相等时，即具有方差齐性，可以直接检验。
~~~
stats.ttest_ind(data1,data2)
~~~
当不确定两总体方差是否相等时，应先利用levene检验，检验两总体是否具有方差齐性。
~~~
stats.levene(data1,data2)
~~~
如果返回结果的p值远大于0.05，那么我们认为两总体具有方差齐性。
如果两总体不具有方差齐性，需要加上参数equal_val并设定为False。如下。
~~~
stats.ttest_ind(data1,data2,equal_var=False)
~~~
### 配对样本T检验(ttest_rel)
~~~
stats.ttest_rel(data1,data2)
~~~
## F检验
~~~
from sklearn.feature_selection.univariate_selection import f_classif
from sklearn.feature_selection import SelectKBest  
from sklearn.feature_selection import chi2
model1 = SelectKBest(f_classif, k=2)#选择k个最佳特征  
model1.fit_transform(data_set[:,1:], data_set[:,0])
print("F检验P值：")
print(model1.pvalues_)
~~~
参考博客：
1. 各种检验例子[https://blog.csdn.net/zlf19910726/article/details/80382481](https://blog.csdn.net/zlf19910726/article/details/80382481)
2. sklearn中f检验的解释[https://blog.csdn.net/jetFlow/article/details/78884619](https://blog.csdn.net/jetFlow/article/details/78884619)
