'''
# 从sklearn.datasets自带的数据中读取糖尿病数据并将其存储在变量diabetes中
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

# 明确特征变量与目标变量
x = diabetes.data
y = diabetes.target

# 从sklearn.model_selection中导入数据分割器
from sklearn.model_selection import train_test_split

# 使用数据分割器将样本数据分割为训练数据和测试数据，其中测试数据占比为20%。
# 数据分割是为了获得训练集和测试集。训练集用来训练模型，测试集用来评估模型性能
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=33,test_size=0.2)

# 从sklearn.linear_model中选用选用线性回归模型LinearRegression来学习数据。
# 我们认为糖尿病数据的特征变量与目标变量之间可能存在某种线性关系，这种线性关系可以用线性回归模型LinearRegression来表达，所以选择改算法进行学习
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化线性回归器
lr = LinearRegression()

# 使用训练数据来估计参数，也就是通过训练数据的学习，为线性回归器找到一组合适的参数，从而获得一个带有参数的、具体的线性回归模型
lr.fit(x_train, y_train)

# 对测试数据进行预测。利用上述训练数据学习得到的带有参数的、具体的线性回归模型对测试数据进行预测
# 即将测试数据中每一条记录的特征变量（例如年龄、性别、体重指数等）输入该线性回归模型中，得到一个该条记录的预测值
lr_y_predict = lr.predict(x_test)

# 模型性能评估。可以通过比较测试数据的模型预测值与真实值之间的差距来评估，例如使用R-squared来评估
from sklearn.metrics import r2_score
print('r2_score: ', r2_score(y_test, lr_y_predict))

'''


# ---------------------------------- 第五章 PCA降维 --------------------------------------
'''
# 利用PCA降维方法对鸢尾花的特征维度（4维）进行降维

## 导入相关库
from sklearn import datasets, decomposition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## 数据输入
### 使用sklearn自带的鸢尾花数据集
### 获取数据
def load_data():
    iris = datasets.load_iris(as_frame=True)
    return iris.data, iris.target

## PCA降维
def test_PCA(*data):
    x, y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(x)
    pd.set_option('display.unicode.east_asian_width', True)
    print('可解释的方差占比：%s' %str(pca.explained_variance_ratio_))

## 可视化
def plot_PCA(*data):
    x, y = data
    pca = decomposition.PCA(n_components=2)
    x_r = pca.fit_transform(x)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1,0,0),(0,1,0),(0,0,1))
    for label, color in zip(np.unique(y), colors):
        position = y == label
        ax.scatter(x_r[position, 0], x_r[position, 1], label = 'target=%d' %label, color = color)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('y[0]')
    ax.legend(loc='best')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax.set_title('PCA降维后样本分布图')
    plt.show()

## 加载数据
x, y = load_data()

## 测试降维效果
test_PCA(x, y)

## 降维结果可视化
plot_PCA(x, y)   

'''

# ---------------------------------- 第六章 凸优化 --------------------------------------
'''
# 导入相关库
import matplotlib.pyplot as plt 
import mpl_toolkits.axisartist as axisartist
import numpy as np

# （1）一元函数的梯度下降
## ① 绘制函数图像
### 获取数据
#### x轴范围为-10～10，且分割为100份
x = np.linspace(-10, 10, 100)
#### 生成一元函数y值
y = x**2+3

### 配置图像
#### 创建画布
fig = plt.figure(figsize=(8, 8))
#### 使用axisartist.Subplot方法创建绘图区对象ax 
ax = axisartist.Subplot(fig, 111)
#### 把绘图区对象添加至画布
fig.add_axes(ax)
#### 使用set_visible方法隐藏绘图区原有所有坐标轴
ax.axis[:].set_visible(False)
#### 使用ax.new_floating_axis添加新坐标轴
ax.axis["x"] = ax.new_floating_axis(0, 0)
#### 给x轴添加箭头
ax.axis["x"].set_axisline_style("->", size=1.0)
#### 使用ax.new_floating_axis添加新坐标轴
ax.axis["y"] = ax.new_floating_axis(1, 0)
#### 给y轴添加箭头
ax.axis["y"].set_axisline_style("->", size=1.0)
#### 设置刻度显示方向，x轴为下方显示，y轴为右侧显示
ax.axis["x"].set_axis_direction("bottom")
ax.axis["y"].set_axis_direction("right")
#### 设置x、y轴范围
plt.xlim(-12, 12)
plt.ylim(-10, 100)

## 绘制并显示图像
plt.plot(x, y)
#plt.show()

## ② 梯度下降过程
### 定义一维函数f(x)=x**2+3的梯度或导数df/dx=2x
def grad_1(x):
    return x*2

### 定义梯度下降函数
def grad_descent(grad, x_current, learning_rate, precision, iters_max):
    for i in range(iters_max):
        print('第', i, '次迭代x值为：', x_current)
        grad_current = grad(x_current)
        if abs(grad_current) < precision:
            break
        else:
            x_current = x_current - grad_current*learning_rate
    print('极小值处x为：', x_current)
    return x_current

# （2）多元函数的梯度下降
## 测试数据
x, y = np.mgrid[-2:2:20j, -2:2:20j]
z = (x**2 + y**2)

## ① 图像绘制并显示
ax = plt.subplot(111, projection='3d')
ax.set_title('f(x,y)=x^2+y^2')
ax.plot_surface(x, y, z, rstride=9, cstride=1, cmap=plt.cm.Blues_r)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#plt.show()

## ② 梯度下降过程
### 定义多元函数 f(x,y)=x**2+y**2 的梯度 f'(x)=2*x, f'(y)=2*y
def grad_2(p):
    derivx = 2 * p[0]
    derivy = 2 * p[1]
    return np.array([derivx, derivy])

### 定义梯度下降函数
def grad_descent(grad, p_current, learning_rate, precision, iters_max):
    for i in range(iters_max):
        print ('第',i,'次迭代p值为:',p_current)   
        grad_current=grad(p_current)
        if np.linalg.norm(grad_current, ord=2)<precision:
            break
        else:
            p_current=p_current-grad_current* learning_rate
    print('极小值处p为：', p_current)
    return p_current


### 赋值并执行
if __name__ == '__main__':
#   grad_descent(grad_1, x_current=5, learning_rate=0.1, precision=0.000001, iters_max=10000)
#   grad_descent(grad_2, p_current=np.array([1, -1]), learning_rate=0.1, precision=0.000001, iters_max=10000)

'''

#---------------------------------- 第七章 线性回归 --------------------------------------
'''
# 房价预测

## 导入相关库
### numpy、pandas
import numpy as np
import pandas as pd
### 数据分割器
from sklearn.model_selection import train_test_split
### 线性回归模型
from sklearn.linear_model import LinearRegression
### 性能评估
from sklearn.metrics import mean_squared_error

## 输入数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
### 明确特征变量和目标变量
x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

## 数据预处理
### 使用数据分割器将样本数据分割为训练数据和测试数据，其中测试数据占比25%。
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33, test_size=0.25)

## 模型训练
### 使用默认配置初始化线性回归器
lr = LinearRegression()
### 使用训练数据来估计参数，也就是通过训练数据的学习，为线性回归器找到一组合适的参数，从而获得一个带有参数的、具体的线性回归模型
lr.fit(x_train, y_train)

## 模型预测
### 对测试数据进行预测。利用上述训练数据学习得到的带有参数的、具体的线性回归模型对测试数据进行预测，即将测试数据中每一条记录的特征变量（例如房间数、不动产税率等）输入该线性回归模型中，得到一个该条记录的预测值
lr_y_predict = lr.predict(x_test)

## 性能评估
print("MSE: ", mean_squared_error(y_test, lr_y_predict))

'''

#---------------------------------- 第八章 逻辑回归 --------------------------------------
'''
# 用逻辑回归分类器来对sklearn自带的乳腺癌数据集进行学习和预测

## 导入相关库
### 数据集
from sklearn.datasets import load_breast_cancer
### 数据分割器
from sklearn.model_selection import train_test_split 
### 数据标准化
from sklearn.preprocessing import StandardScaler
### 学习模型
from sklearn.linear_model import LogisticRegression
### 性能评估
from sklearn.metrics import classification_report


## 输入数据
### 加载数据集
breast_cancer = load_breast_cancer()
### 分离出特征变量和目标变量
x = breast_cancer.data
y = breast_cancer.target

## 数据预处理
### 使用数据分割器将样本数据分割为训练数据和测试数据
x_train, x_test, y_train,  y_test = train_test_split(x, y, random_state=33, test_size=0.3)
### 对数据进行标准化处理，使得每个特征维度的均值为0，方差为1，防止受到某个维度特征数值较大的影响
breast_cancer_ss = StandardScaler()
x_train = breast_cancer_ss.fit_transform(x_train)
x_test = breast_cancer_ss.transform(x_test)

## 模型训练
### 从sklearn.linear_model中选用逻辑回归模型LogtisticRegression来学习
#### 使用默认配置初始化逻辑线性回归器
LR = LogisticRegression()
#### 进行训练
LR.fit(x_train, y_train)
#### 结果预测
LR_y_predict = LR.predict(x_test)

## 性能评估
print("Accuracy: ", LR.score(x_test, y_test))
print(classification_report(y_test, LR_y_predict, target_names=['benign', 'maligant']))

'''