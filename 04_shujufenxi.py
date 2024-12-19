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
