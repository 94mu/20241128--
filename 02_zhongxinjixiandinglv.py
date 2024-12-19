import numpy as np
import matplotlib.pyplot as plt

#-------------- 总体情况 -------------------
# 抛掷骰子
random_data = np.random.randint(1, 7, 60000)
# print(random_data)

# 抛掷结果的统计值
print("均值为：", random_data.mean())
print("标准差为：", random_data.std())

# 直方图展示抛掷骰子的结果
# plt.hist(random_data, bins=50)

# 结果展示
# plt.show()

# ------------ 单次抽样情况 ------------------
# 从总体数据中抽取100次抛掷结果
sample_100 = []

for i in range(0, 100):
    sample_100.append(random_data[int(np.random.random() * len(random_data))])

# 输出结果
# print(sample_100)

# 展示此次抽样结果的统计值
# print("样本均值为：", np.mean(sample_100))
# print("样本标准差为：", np.std(sample_100, ddof = 1))

# ----------- 多次抽样情况 ------------
# 抽取50000组，每组抽取100个结果
sample_100_mean = []

for i in range(0, 50000):
    sample = []
    for j in range(0, 100):
        sample.append(random_data[int(np.random.random() * len(random_data))])
    sample_100_mean.append(np.mean(sample))

sample_many_mean = np.mean(sample_100_mean)
print("多次抽取样本的总均值：", sample_many_mean)
plt.hist(sample_100_mean, bins=200)
plt.show()