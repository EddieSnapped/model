import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats
from scipy import stats
matplotlib.use('Qt5Agg')

# 创建一个示例 DataFrame
def analyze_1d(df: pd.DataFrame, name):

    # 描述性统计
    desc_stats = df[name].describe()
    print(desc_stats)


    # 设置绘图风格
    sns.set(style="whitegrid")

    # 直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df[name], kde=False, bins=10, color='skyblue')
    plt.title('Histogram of Values')
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.show()

    # 箱型图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[name], color='lightgreen')
    plt.title('Boxplot of Values')
    plt.xlabel(name)
    plt.show()

    # 密度图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[name], shade=True, color='red')
    plt.title('Density Plot of Values')
    plt.xlabel(name)
    plt.ylabel('Density')
    plt.show()



    # 计算 Z-Score
    z_scores = stats.zscore(df[name])

    # 设置阈值，通常 Z-Score > 3 被认为是异常值
    outliers = df[name][abs(z_scores) > 3]
    print("异常值：", outliers)


    # Shapiro-Wilk 正态性检验
    stat, p_value = stats.shapiro(df[name])
    print(f"Shapiro-Wilk test statistic: {stat}, p-value: {p_value}")

    if p_value > 0.05:
        print("数据符合正态分布")
    else:
        print("数据不符合正态分布")
