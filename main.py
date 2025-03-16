import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')  # 使用 TkAgg 后端
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

complete_data = np.column_stack([data, target])

columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

boston = pd.DataFrame(complete_data, columns=columns)

# 数据集信息
boston.info

boston.describe()

# boston.hist(bins=20, figsize=(20,15))
# plt.show()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

correlation_matrix = boston.corr()

#
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
# plt.show()


# plt.scatter(boston['LSTAT'], boston['MEDV'], alpha=0.5)
# plt.xlabel("地位较低人群的百分比（RM）")
# plt.ylabel("房价中位数（MEDV）")
# plt.title("LSTAT VS MEDV")
# plt.show()

boston.boxplot(column=['RM'])
# plt.show()

boston.loc[boston['RM'] > 8, 'RM'] = 8

x = boston.drop(columns='MEDV', axis=1)
y = boston['MEDV']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# 计算预测值和真实值之间的均方误差: 越小越好
mse = mean_squared_error(y_test, y_pred)

# 决定系数：模型越好R^2 -> 1; 模型越差 R^2 -> 0
r2 = r2_score(y_test, y_pred)

print(f"均方误差（MSE）：{np.sqrt(mse)}")
print(f"决定系数（R^2）：{r2}")

plt.scatter(y_test, y_pred)
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title('实际价格 VS 预测价格')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.show()
