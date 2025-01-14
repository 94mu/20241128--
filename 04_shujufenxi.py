# -------------------------------------------- 二、NumPy基础 -----------------------------------------------
"""
## 导入numpy库
import numpy as np

## 2.1 创建数组
### 2.1.1 np.arr(列表或元组)
'''
#### [1 2 3 4 5]
arr = np.array([1, 2, 3, 4, 5]) 
arr = np.array((1, 2, 3, 4, 5)) 
'''
### 2.1.2 np.arange(start, end, step)
'''
#### np.arange()和range()非常相似，但：range()的步长只能是整数，而np.arange()的步长可以是任意数。
#### np.arange()只能创建一维数组
#### [0 1 2 3 4 5 6 7 8 9]
arr = np.arange(10) 
arr = np.arange(0, 10)
arr = np.arange(0, 10, 1)
#### [1.5 3.5 5.5 7.5 9.5]
arr = np.arange(1.5, 10.5, 2)
'''
### 2.1.3 np.linspace(start, end, num, endpoint=True或False)
'''
arr = np.linspace(0, 10, 20)
arr = np.linspace(0, 10, 20, endpoint=False)
'''
### 2.1.4 np.zeros((a, b, ..., n), dtype=int或float) 和 np.ones((a, b, ..., n), dtype=int或float)
'''
#### 默认情况下，得到的数组元素都是float类型，可以使用dtype=int将数组元素定义成int类型
arr1 = np.zeros([3, 3])
arr1 = np.zeros((3, 3))
arr1 = np.zeros(shape=(3, 3))

arr2 = np.ones([3, 3])
arr2 = np.ones((3, 3))
arr2 = np.ones(shape=(3, 3))
'''
### 2.1.5 np.random.randint(start, end, size=元组或整数)
'''
#### 一维
arr = np.random.randint(10, 20, size=(5))
arr = np.random.randint(10, 20, size=5)
arr = np.random.randint(10, 20, 5)
#### 二维
arr = np.random.randint(10, 20, size=(2, 5))
arr = np.random.randint(10, 20, (2, 5))
'''
### 2.1.6 np.random.rand(m, n)
'''
#### 元素都是[0,1)内的浮点数
#### 一维
arr = np.random.rand(5)
#### 二维
arr = np.random.rand(2, 5)
'''
### 2.1.7 np.random.randn(m, n)
'''
#### 一维
arr = np.random.randn(5)
#### 二维
arr = np.random.randn(2, 5)
'''

## 2.2 数组属性
'''
### 一维
nums_1 = [1, 2, 3, 4, 5, 6, 7, 8]
arr_1 = np.array(nums_1)
### 二维
nums_2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr_2 = np.array(nums_2)
### 三维
nums_3 =  [[[1, 2, 3], [1, 5, 6]], [[1, 3, 3], [4, 5, 6]]]
arr_3 = np.array(nums_3)
'''
### 2.2.1 维度 ndim
'''
arr_1.ndim # 1
arr_2.ndim # 2
arr_3.ndim # 3
'''
### 2.2.2 形状 shape
'''
arr_1.shape # (8,)
arr_2.shape # (2, 4)
arr_3.shape # (2, 2, 3)
'''
### 2.2.3 元素个数 size
'''
arr_1.size # 8
arr_2.size # 8
arr_3.size # 12
'''
### 2.2.4 元素类型 dtype
'''
arr_1.dtype # int32
arr_2.dtype # int32
arr_3.dtype # int32
'''

## 2.3 元素操作
### 2.3.1 访问元素
'''
#### 一维数组
nums = [3, 9, 1, 12, 50, 21]
arr = np.array(nums)
arr[0] # 3
#### 二维数组
nums = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]
arr = np.array(nums)
arr[1][2] # 6
arr[1, 2] # 6
#### 负数下标
nums = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]
arr = np.array(nums)
arr[-1][-1] # 10
arr[-1][-2] # 8
'''
### 2.3.2 修改元素
'''
arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
arr[2][2] = 99        # [[10, 20, 30], [40, 50, 60], [70, 80, 99]]
arr[1] = [44, 55, 66] # [[10, 20, 30], [44, 55, 66], [70, 80, 99]]
'''
### 2.3.3 添加元素 np.append(arr, value, axis=n)
'''
#### 一维数组
arr = np.array([1, 2, 3, 4])
np.append(arr, 5) # [1 2 3 4 5]
#### 二维数组（axis=0）
arr = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
np.append(arr, [[11, 22, 33, 44]], axis=0) # [[10 20 30 40] [50 60 70 80] [11 22 33 44]]
#### 二维数组（axis=1）
arr = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
np.append(arr, [[11, 22], [33, 44]], axis=1) # [[10 20 30 40 11 22] [50 60 70 80 33 44]]
#### 二维数组（axis为空）
##### append()会直接将两个数组“打平”成一维数组，然后进行合并
arr = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
np.append(arr, [11, 22, 33, 44])     # [10 20 30 40 50 60 70 80 11 22 33 44]
np.append(arr, [[11, 22, 33, 44]])   # [10 20 30 40 50 60 70 80 11 22 33 44]
np.append(arr, [[11, 22], [33, 44]]) # [10 20 30 40 50 60 70 80 11 22 33 44]
'''
### 2.3.4 删除元素 np.delete(arr, m, axis=n)
'''
#### 一维数组
arr = np.array([10, 20, 30, 40, 50])
np.delete(arr, 2) # [10 20 40 50]
#### 二维数组
#### 在NumPy中，对二维数组进行删除操作时，一般是删除一整行或一整列，很少会去删除某一个元素。
##### axis=0 删除行
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.delete(arr, 2, axis=0) # [[1 2 3] [4 5 6]]
##### axis=1 删除列
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.delete(arr, 2, axis=1) # [[1 2] [4 5] [7 8]]
'''
### 2.3.5 切片
'''
#### 一维数组
arr = np.arange(10, 20, 2) # [10 12 14 16 18]
arr[1:3] # [12 14]
#### 二维数组
arr = np.arange(10, 30).reshape(5, 4) # [[10 11 12 13] [14 15 16 17] [18 19 20 21] [22 23 24 25] [26 27 28 29]]
arr[0, :]     # [10 11 12 13]
arr[:, 0]     # [13 17 21 25 29]
arr[0, 1:3]   # [11 12]
arr[0:2, 0:2] # [[10 11] [14 15]]
#### 多维数组
arr = np.arange(10, 22).reshape(2, 3, 2) # [[[10 11] [12 13] [14 15]] [[16 17] [18 19] [20 21]]]
arr[0, ...] # [[10 11] [12 13] [14 15]]
arr[..., 0] # [[10 12 14] [16 18 20]]
'''

## 2.4 数组操作
## arr必须由numpy创建
### 2.4.1 修改形状 arr.reshape(m, n)
'''
#### 一维数组
arr1 = np.arange(10)
arr2 = arr1.reshape(2, 5) # [[0 1 2 3 4] [5 6 7 8 9]]
#### 二维数组
arr1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
arr2 = arr1.reshape(4, 2) # [[1 2] [3 4] [5 6] [7 8]]
'''
### 2.4.2 修改维度 np.array(arr, ndmin=n) 
'''
#### 低维转高维
arr = np.array([1, 2])
np.array(arr, ndmin=2) # [[1] [2]]
np.array(arr, ndmin=3) # [[[1 2]]]
#### 高维转低维
arr = np.array([[1, 2], [3, 4]])
# ndmin参数本质上是用于指定数组的最小维度，ndmin=1表示数组的最小维度为1（大于或等于1都可以）。这里arr的维度为2，本身就已经大于1了，所以ndmin=1并不会将二维数组转换为一维数组。
np.array(arr, ndmin=1) # [[1 2] [3 4]] 
'''
### 2.4.3 翻转数组 np.transpose(arr)
'''
arr1 = np.arange(10, 25).reshape(5, 3) # [[10 11 12] [13 14 15] [16 17 18] [19 20 21] [22 23 24]]
arr2 = np.transpose(arr1) # [[10 13 16 19 22] [11 14 17 20 23] [12 15 18 21 24]]
'''
### 2.4.4 数组去重 np.unique(arr)
'''
# 不管是多少维的数组，unique()返回的都是一个一维数组。
#### 一维数组
arr = np.array([21, 10, 32, 10, 45, 10, 32, 60])
np.unique(arr)  # [10 21 32 45 60]
#### 二维数组
arr = np.array([[21, 10, 32, 10], [45, 10, 32, 60]])
np.unique(arr) # [10 21 32 45 60]
'''
### 2.4.5 合并数组 np.concatenate((arr1, arr2), axis=n)；np.stack((arr1, arr2), axis=n)
'''
#### （1）沿“现有轴”合并，指的是根据数组原有的轴进行合并，合并前后的两个数组维度相同。
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
##### 沿纵轴合并
np.concatenate((arr1, arr2), axis=0) # [[1 2]  [3 4] [5 6] [7 8]]
np.vstack((arr1, arr2))              # [[1 2]  [3 4] [5 6] [7 8]]
##### 沿横轴合并
np.concatenate((arr1, arr2), axis=1) # [[1 2 5 6] [3 4 7 8]]
np.hstack((arr1, arr2))              # [[1 2 5 6] [3 4 7 8]]
#### （2）沿“新的轴”合并，指的是创建一个新的轴进行合并，合并后的数组维度更大。
np.stack((arr1, arr2), axis=0)  # [[[1 2]  [3 4]] [[5 6]  [7 8]]]
np.stack((arr1, arr2), axis=1)  # [[[1 2]  [5 6]] [[3 4]  [7 8]]]
'''
### 2.4.6 分割数组 np.split(arr, section=整数或数组, axis=n)
'''
### split()会返回一个列表，该列表的每一个元素都是一个数组
#### 一维数组
arr = np.arange(6)
result = np.split(arr, 3) # [array([0,1]), array([2,3]), array([4,5])]
#### 二维数组
arr = np.arange(8).reshape(4, 2)   # [[0 1] [2 3] [4 5] [6 7]]
#### 沿纵轴分割
result1 = np.split(arr, 2, axis=0) # [array([0,1], [2,3]), array([4,5], [6,7])]
np.vsplit(arr, 2)
#### 沿横轴分割
result2 = np.split(arr, 2, axis=1) # [array([0], [2], [4], [6]), array([1], [3], [5], [7])]
np.hsplit(arr, 2)
'''

## 2.5 各种运算
## 在实际开发中，更推荐使用运算符的方式来进行计算，主要是这种方式更加简单方便
## 数组的运算是“元素级”的。所谓“元素级”，指运算是针对每一个元素进行的
### 2.5.1 基本运算（加、减、乘、除、求余、求幂）
'''
arr1 = np.array([1, 2])
arr2 = np.array([3, 4])
#### 加
arr1 + arr2                # [4 6]
np.add(arr1, arr2)         # [4 6]
#### 减
arr1 + arr2                # [-2 -2]
np.subtract(arr1, arr2)    # [-2 -2]
#### 乘
arr1 * arr2                # [3 8]
np.multiply(arr1, arr2)    # [3 8]
#### 除 
arr1 / arr2                # [0.33333333 0.5]
np.divide(arr1, arr2)      # [0.33333333 0.5]
#### 求余
arr1 % arr2                # [1 2]
np.mod(arr1, arr2)         # [1 2]
#### 求幂
arr1 ** arr2               # [1 16]
np.power(arr1, arr2)       # [1 16]
'''
### 2.5.2 比较运算（大于、小于、大于等于、小于等于、等于、不等于）
'''
#### 一维数组
a = np.array([1, 2])
b = np.array([3, 4])
a > b # [False False]
a < b # [True True]
#### 二维数组
a = np.array([[32, 21], [15, 43]])
b = np.array([[16, 18], [27, 36]])
a > b # [[Ture True] [False True]]
a < b # [[False False] [True False]]
'''
### 2.5.3 标量运算
'''
### 数组的标量运算可以简单理解为将一个数组和一个数进行加、减、乘、除等操作
### 不管是多少维的数组，标量运算都是对数组中每一个元素进行相同的操作
#### 一维数组
arr = np.array([10, 20, 30])
arr + 10  # [20 30 40]
arr - 10  # [ 0 10 20]
arr * 2   # [20 40 60]
arr / 10  # [1. 2. 3.]
#### 二维数组
arr = np.array([[10, 20, 30], [40, 50, 60]])
arr + 10  # [[20 30 40] [50 60 70]]
arr - 10  # [[ 0 10 20] [30 40 50]]
arr * 2   # [[ 20  40  60] [ 80 100 120]]
arr / 10  # [[1. 2. 3.] [4. 5. 6.]]
'''
### 2.5.4 数学函数（平方、绝对值、平方根、四舍五入、向上取整、向下取整、求导数、求三角函数值等）
'''
### 会对数组中的每一个元素进行操作
#### np.square()
#### np.abs()
#### np.sqrt()
#### np.around()
arr = np.array([[1.2, 2.7], [3.5, 4.3]])
np.around(arr) # [[1. 3.] [4. 4.]]
#### 向上取整
arr = np.array([3.0, 0.4, 0.6, -1.1, -1.9])
np.ceil(arr) # [ 3.  1.  1. -1. -1.]
#### 向下取整
arr = np.array([3.0, 0.4, 0.6, -1.1, -1.9])
np.floor(arr) # [ 3.  0.  0. -2. -2.]
'''
### 2.5.5 统计函数（求和、最大值、最小值、中位数、平均数、加权平均数、方差、标准差、百分位数、众数）
'''
#### np.sum(arr, axis=n)
#### np.max(arr, axis=n)
#### np.min(arr, axis=n)
#### np.median(arr, axis=n)
#### np.mean(arr, axis=n)
#### np.average(arr, axis=n)
#### np.var(arr, axis=n)
#### np.std(arr, axis=n)
#### np.percentile(arr, q, axis=n)
#### np.argmax((np.bincount(arr)) 极少使用
'''

## 2.6 遍历数组
### 2.6.1 一维数组
'''
arr = np.array([10, 36, 43, 25, 27])
for item in arr:    
    print(item)  
'''
### 2.6.2 二维数组
'''
### 对于二维数组或更多维数组来说，使用for循环遍历时，默认情况下是针对第一个轴（axis=0）来进行的。
arr = np.array([[1, 2], [3, 4], [5, 6]])
for item in arr:    
    print(item)
### 如果要遍历数组的所有元素，则需要使用数组的flat属性。
### arr.flat会“打平”arr，从而得到一个包含1、2、3、4、5、6的对象。它并不是一个数组，如果想要将其转换为数组，可以使用np.array(arr.flat)来实现。
### arr.flat本身返回的是一个可迭代对象，我们不需要再多此一举将其转换为数组，而是可以直接使用for循环来对其进行遍历。
arr = np.array([[1, 2], [3, 4], [5, 6]])
for item in arr.flat:    
    print(item)
'''

## 2.7 大小排序
## 在NumPy中，可以使用sort()来对数组进行排序。由于数组对象和NumPy都有sort()，所以需要分成以下两种情况来考虑
### 如果是在“数组对象”上调用sort()，则会对数组本身进行排序
### 如果是在“NumPy”上调用sort()，则会返回该数组对象的排序副本
### 2.7.1 数组对象调用sort()
### 默认情况下，sort()实现的是升序排列，也就是从小到大排序。
'''
arr = np.array([22, 40, 36, 15, 10])
arr.sort() # [10 15 22 36 40]
### 如果想要实现数组的降序排列，可以使用一种变通的方式
arr = np.array([22, 40, 36, 15, 10])
arr.sort()
result = arr[::-1] # [40 36 22 15 10]
'''
### 2.7.2 numpy调用sort()
'''
### arr.sort()会修改原数组，np.sort(arr)不会修改原数组。在实际开发中，更推荐使用np.sort(arr)这种方式。
arr = np.array([22, 40, 36, 15, 10])
result = np.sort(arr) # [10 15 22 36 40]
'''
"""

# -------------------------------------------- 三、NumPy进阶 -----------------------------------------------
"""
## 导入numpy库
import numpy as np

## 3.1 浅拷贝和深拷贝
'''
### 3.1.1 浅拷贝 arr.view()
### 3.1.2 深拷贝 arr.copy()
'''

## 3.2 axis的值

## 3.3 广播机制

## 3.4 读写文件
### 3.4.1 读取文件 np.loadtxt(path, delimiter="分隔符")
### 读取出来的数据是浮点型数据。如果想要使得数据是整型数据，可以使用dtype=int来实现
'''
arr = np.loadtxt(r"data/test.csv", delimiter=",", dtype=int)
'''
### 3.4.2 写入文件 np.savetxt(path, arr, delimiter="分隔符")
### 可以使用fmt="%d"使得数据都是整型数据
'''
arr = np.arange(9).reshape(3, 3)
np.savetxt(r"data/arr.csv", arr, delimiter=",", fmt="%d")
'''

### 3.5 矩阵
### 在NumPy中，矩阵和数组是两种不同的数据类型。虽然它们很相似，但是有着本质上的区别。数组是一个ndarray对象，矩阵是一个matrix对象。
### 矩阵都是二维的，可以把它看成一种特殊的二维数组。NumPy官方建议，如果在某种情况下矩阵和数组都适用，那就选择数组。因为数组更加灵活，速度更快。
### 对于矩阵，读者只需要了解一下基本创建方式就可以了，其他的不用过多了解，因为大多数情况下使用二维数组就可以完成工作。
### 在NumPy中，不管是哪一种矩阵，都是使用matlib模块的方法来创建的。
### 默认情况下，矩阵元素类型是浮点型，可以使用dtype=int将其类型定义成整型。
'''
#### 3.5.1 全0矩阵  np.matlib.zeros((m, n))
#### 3.5.2 全1矩阵  np.matlib.ones((m, n))
#### 3.5.3 单位矩阵 np.matlib.identity(m)
#### 3.5.4 随机矩阵 np.matlib.rand(m, n) 矩阵的每一个元素的值都在0～1内
'''
"""

# -------------------------------------------- 四、Pandas简介 -----------------------------------------------
import pandas as pd


## 4.2 Series
## Series可以看成一种特殊的一维数组
### 4.2.1 创建Series
'''
pd.Series(data, index=列表)
#### （1）当data是一个列表时
##### 不指定index的值
se = pd.Series(["红", "绿", "蓝"])
##### 指定index的值
se = pd.Series(["红", "绿", "蓝"], index=["red", "green", "blue"])
#### （2）当data是一个字典
se = pd.Series({"小杰": 1001, "小红": 1002, "小明": 1003})
'''
### 4.2.2 Series属性
'''
se = pd.Series(["红", "绿", "蓝"], index=["red", "green", "blue"])
se.index   # 行名
se.values  # 数据
'''
### 4.2.3 获取某行的值
'''
se = pd.Series({"小杰": 1001, "小红": 1002, "小明": 1003})
result = se.loc["小杰"]
'''
### 4.2.4 深入了解
'''
➤ 字典是无序的，而Series是有序的。
➤ 字典的key是不可变的，而Series的index是可变的。
➤ Series提供了大量的统计方法，但是字典没有。
'''

## 4.3 DataFrame
### 4.3.1 创建DataFrame
'''
pd.DataFrame(data, index=列表, columns=列表)
#### （1）data是一个列表
#####     ① 列表的元素是列表
data = [["小杰", "男", 20],    
        ["小红", "女", 19],    
        ["小明", "男", 21]]
df = pd.DataFrame(data)
#####      指定行名和列名
data = [["小杰", "男", 20],    
        ["小红", "女", 19],    
        ["小明", "男", 21]]
df = pd.DataFrame(data, index=["s1", "s2", "s3"], columns=["name", "gender", "age"])
df = pd.DataFrame(data, index=["学生1", "学生2", "学生3"], columns=["姓名", "性别", "年龄"])
#####     ② 列表的元素是字典
data = [{"name": "小杰", "gender": "男", "age": 20},    
        {"name": "小红", "gender": "女", "age": 19},    
        {"name": "小明", "ender": "男", "age": 21}]
df = pd.DataFrame(data)
#####       指定行名
df = pd.DataFrame(data, index=["s1", "s2", "s3"])
#### （2）data是一个字典
#### 当data是一个字典时，字典的key会作为DataFrame的列名，字典的value会作为DataFrame的数据。需要注意的是，字典的每一个value都要求是一个列表或者一个值。
#####     ① value是一个列表
data = {"name": ["小杰", "小红", "小明"], "gender": ["男", "女", "男"], "age": [20, 19, 21]}
df = pd.DataFrame(data, index=["s1", "s2", "s3"])
#####     ② value是一个值
##### 如果字典的某一个value是一个值，那么这个值就会映射到每一行中。
##### 对于data是一个字典这种情况，在实际开发中用得比较少，读者简单了解一下就可以了
data = {"name": ["小杰", "小红", "小明"], "gender": ["男", "女", "男"], "age": [20, 19, 21], "class": "一班"}
df = pd.DataFrame(data, index=["s1", "s2", "s3"])
'''

### 4.3.2 DataFrame的属性
'''
### dtypes、values、index、columns、shape、size 
data = [["小杰", "男", 20],    
        ["小红", "女", 19],    
        ["小明", "男", 21]]
df = pd.DataFrame(data, index=["s1", "s2", "s3"], columns=["name", "gender", "age"])
#### 系统会把DataFrame每一列的类型分别列举出来。这里可以看出DataFrame非常重要的一个特点：不同列的数据类型可以不一样，但是同一列的数据类型一般要求相同。
#### DataFrame的值会以数组的形式返回
df.dtypes
df.values
#### 获取行名和列名
df.index
df.columns
#### 获取行数和列数
#### df.shape返回的是一个元组，第1个元素是行数，第2个元素是列数。查看行名和列名，以及查看行数和列数等操作在实际工作中会经常用到。
df.shape
df.shape[0]
df.shape[1]
#### 元素的个数
df.size
'''

### 4.3.3 深入了解
#### （1）文件格式
'''
#### 当data是一个列表时，如果想将DataFrame保存到文件中，一般是保存到CSV文件中
#### 当data是一个字典时，如果想将DataFrame保存到文件中，一般是保存到JSON文件中
'''
#### （2）对齐输出
'''
#### 解决这种输出内容不对齐的问题，可以在print()语句的前一行加上下面这句代码：
pd.set_option("display.unicode.east_asian_width", True)
'''
#### （3）不需要指定行名
'''
#### 在实际工作中，创建DataFrame时，一般需要指定列名，但不需要指定行名。如果不显式指定行名，那么它就是从0开始的连续整数。
'''

# ------------------------ 读取数据库 ---------------------------
'''
import pymysql
import pandas as pd

# 第1步：连接数据库conn = pymysql
conn = pymysql.connect(host="localhost",
                       port=3306,
                       user="root", 
                       password=".666MySQL888.",  
                      db="xiaomuwu",
                       charset="utf8")

# 第2步：读取数据
df = pd.read_sql("select * from student", conn)

# 第3步：关闭数据库
conn.close()

pd.set_option("display.unicode.east_asian_width", True)
print(df)
'''
# ------------------------ 处理天气数据 ----------------------------
'''
import pandas as pd

df = pd.read_csv(r"data/guangzhou.csv")

# 操作“月份”列
df["月份"] = df["月份"].astype(str) + "月"

# 操作“温度”列
df["温度"] = df["温度"].astype(str) + "℃"

# 设置“月份”列为行名
df = df.set_index("月份")

pd.set_option("display.unicode.east_asian_width", True)
print(df)
'''
# ------------------------ 拆分数据 ----------------------------
'''
#import pandas as pd 

df = pd.read_csv(r"data/journey.csv")

# 拆分列
temp = df["from_to"].str.split("_", expand = True)

# 修改列名
temp.columns = ["from", "to"]

# 合并temp到df
df = df.join(temp)

# 删除原始列
df = df.drop("from_to", axis=1)

pd.set_option("display.unicode.east_asian_width", True)

print(df)
'''
# ------------------------ 课后习题7.1 ----------------------------
#下面有一个DataFrame，请将vip这一列的yes、no分别替换成True、False。
'''
import pandas as pd

data = [
    ["小杰", "yes"],    
   ["小红", "yes"],    
    ["小明", "no"],    
   ["小华", "no"],    
    ["小莉", "yes"]]

df = pd.DataFrame(data, columns=["name", "vip"])

df.replace({"yes":"True", "no":"False"})
df.replace(["yes", "no"], ["True", "False"])

pd.set_option("display.unicode.east_asian_width", True)
print(df)
'''
# ------------------------ 课后习题7.2 ----------------------------
#下面有一个DataFrame，请将price这一列的“元”去掉，然后将这一列的类型转换成浮点型（注意这一列本身类型是字符串型）
'''
import pandas as pd

data = [
    ["苹果", "6.4元", "秋季"],    
    ["西瓜", "2.5元", "夏季"],    
    ["香蕉", "7.8元", "四季"],    
    ["李子", "12.4元", "夏秋"],    
    ["芒果", "3.5元", "夏季"]]

df = pd.DataFrame(data, columns=["fruit", "price", "season"])

df["price"] = df["price"].str.replace("元", "")
df["price"] = df["price"].astype(float)

pd.set_option("display.unicode.east_asian_width", True)
print(df)
'''
# ------------------------ 课后习题7.3 ----------------------------
# 下面有一个DataFrame，请把name这一列的“姓”和“名”的第一个字母都转换成大写字母。例如，第1行的“bill gates”要转换成“Bill Gates”。
'''
import pandas as pd

data = [
    ["bill gates", 1955],    
    ["steve jobs", 1955],    
    ["tim cook", 1960],    
    ["elon musk", 1971],    
    ["larry page", 1973]]
df = pd.DataFrame(data, columns=["name", "birth"])

def change(item):
    result = ""
    for i in range(len(item)):
        # 如果是整个字符串第1个字符
        if i == 0:
            result += item[i].upper()
        else:
            # 如果前一个字符是一个空格
            if item[i-1] == " ":
                result += item[i].upper()
            else:
                result += item[i]
    return result

df["name"] = df["name"].map(change)

pd.set_option("display.unicode.east_asian_width", True)
print(df)
'''
# ------------------------ 课后习题7.4 ----------------------------
# 下面有一个DataFrame，请找出成绩排名前3的数据，并且将其输出
'''
import pandas as pd

data = [    
    [1001, "小杰", "男", 650, "一班"],    
    [1002, "小红", "女", 645, "一班"],    
    [1003, "小明", "男", 590, "二班"],    
    [1004, "小华", "男", 640, "二班"],    
    [1005, "小莉", "女", 635, "二班"]]

df = pd.DataFrame(data, columns=["学号", "姓名", "性别", "成绩", "班级"])

df.sort_values(by=["成绩"], ascending=False)

result = df.head(3)

pd.set_option("display.unicode.east_asian_width", True)
print(result)
'''

# ---------------------------------- 第九章 时间序列 --------------------------------------
"""
## 导入库
import pandas as pd

## 输入数据
''' data_1_3
data_1_3 = [["20220101", 197, 165, 156, 200],    
            ["20220102", 161, 164, 155, 188],    
            ["20220103", 146, 214, 100, 215],    
            ["20220104", 228, 233, 215, 236],    
            ["20220105", 230, 226, 220, 231]]

df = pd.DataFrame(data1_3, columns = ["日期", "开盘", "收盘", "最低", "最高"])
'''
''' data_4
data_4 = [["20220101", 197, 165, 156, 200],    
          ["20220131", 161, 164, 155, 188],    
          ["20220201", 146, 214, 100, 215],    
          ["20220228", 228, 233, 215, 236],    
          ["20220301", 230, 226, 220, 231],
          ["20220331", 222, 229, 219, 230]]
df = pd.DataFrame(data_4, columns = ["日期", "开盘", "收盘", "最低", "最高"])
'''
''' data_5__1
data_5__1 = [["20220101", 197, 165, 156, 200],    
             ["20220102", 161, 164, 155, 188],    
             ["20220103", 146, 214, 100, 215],    
             ["20220104", 228, 233, 215, 236],    
             ["20220105", 230, 226, 220, 231],    
             ["20220106", 222, 229, 219, 230],    
             ["20220107", 172, 178, 161, 190],    
             ["20220108", 210, 225, 209, 230],    
             ["20220109", 221, 218, 216, 229],    
             ["20220110", 235, 227, 223, 238]]
df = pd.DataFrame(data_5__1, columns = ["日期", "开盘", "收盘", "最低", "最高"])
'''
''' data_5__2
data_5__2 = [["20220101", 193, 201, 175, 212],    
             ["20220108", 222, 223, 216, 232]]
df = pd.DataFrame(data_5__2, columns = ["日期", "开盘", "收盘", "最低", "最高"])
'''
''' data_6
data_6 = [["2022-01-01", 274],    
        ["2022-01-02", 301],    
        ["2022-01-03", 295],    
        ["2022-01-04", 312],    
        ["2022-01-05", 347]]
df = pd.DataFrame(data_6, columns=["日期", "order"])
'''
''' data_7__1
data_7__1 = [[1001, "小杰", "男", 650, "一班"],    
             [1002, "小红", "女", 645, "一班"],    
             [1003, "小明", "男", 590, "二班"],    
             [1004, "小华", "男", 635, "二班"],    
             [1005, "小莉", "女", 640, "二班"]]
df = pd.DataFrame(data_7__1, columns=["学号", "姓名", "性别", "成绩", "班级"])
'''
''' data_7__2
data_7__2 = [["20220101", 197, 165, 156, 200],    
             ["20220102", 161, 164, 155, 188],    
             ["20220103", 146, 214, 100, 215],    
             ["20220104", 228, 233, 215, 236],    
             ["20220105", 230, 226, 220, 231],    
             ["20220106", 222, 229, 219, 230],    
             ["20220107", 172, 178, 161, 190],    
             ["20220108", 210, 225, 209, 230],    
             ["20220109", 221, 218, 216, 229],    
             ["20220110", 235, 227, 223, 238]]
df = pd.DataFrame(data_7__2, columns = ["日期", "开盘", "收盘", "最低", "最高"])
'''

df = pd.read_csv(r"data/sales.csv")

## （1）将字符串转换为时间对象 并 设置行名
df["日期"] = pd.to_datetime(df["日期"])
df= df.set_index("日期")

## （2）查看类型
### 转换后
#### print(df.index.dtype)
#### print(type(df.index))

## （3.1）获取年份,并增加一列
### df["年份"] = df.index.year
## （3.2）获取星期几，增加一列，并作替换处理
### df["星期"] = df.index.weekday
### df["星期"] = df["星期"].replace({0: "星期一", 1: "星期二", 2: "星期三", 3: "星期四", 4: "星期五", 5: "星期六", 6: "星期日"})

## （4）时间切片
### result = df["2022-01-01":"2022-01-03"]

## （5.1）降采样
### result = df.resample("2D").mean()
## （5.2.1）升采样 - 向前填充
### result = df.resample("D").ffill()
## （5.2.2）升采样 - 向后填充
### result = df.resample("D").bfill()

## （6）移动计算
### result = df.rolling(3).sum()

## （7.1）分组器 - 使用参数key
### groups = df.groupby(pd.Grouper(key="班级"))
### result = groups.get_group("一班")
## （7.2）分组器 - 使用参数freq
### result = df.groupby(pd.Grouper(freq="5D"))
### for name, group in result:
###     pd.set_option("display.unicode.east_asian_width", True)
###     print(name)
###     print(group)

## （8.1）求每个月的销量总和
### result = df.resample("M").sum()
### result.columns = ["销量总和"]
### result.index = [str(x) + "月" for x in range(1, 13)]
## （8.2）求每个月最高销量对应的日期
groups = df.groupby(pd.Grouper(freq="ME"))
result = groups.idxmax()
result.index = [str(x)+"月" for x in range(1, 13)]
result.columns = ["销量最高的一天"]
result.loc[:, "销量"] = list((df.resample("M").max())["销售额"])

## 输出结果
pd.set_option("display.unicode.east_asian_width", True)
### print(df)
print(result)

"""


## ------------------------ 求每个月的销量总和 ----------------------------
'''
import pandas as pd

### 读取文件
df = pd.read_csv(r"data/sales.csv")

### 将字符串转换为时间对象 并 设置行名
df["日期"] = pd.to_datetime(df["日期"])
df = df.set_index("日期")

### 降采样
result = df.resample("M").max()

### 修改列名
#result.columns = ["销量总和"]

### 输出
pd.set_option("display.unicode.east_asian_width", True)
print(result)
'''
