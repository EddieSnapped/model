import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
import warnings
# warnings.filterwarnings('ignore')

import numpy as np


def entropy_weight(data):
    """
    计算熵权法权重
    :param data: 二维数组，每行代表一个样本，每列代表一个指标
    :return: 各指标对应的权重向量
    """
    # 归一化数据，每行数据之和为1
    data = data / np.sum(data, axis=0)

    # 处理可能存在的0值（避免log(0)）
    data = np.where(data == 0, 1e-8, data)

    # 计算信息熵
    num_samples, num_features = data.shape
    entropy = - np.sum(data * np.log(data), axis=0) / np.log(num_samples)

    # 计算差异系数
    difference_coefficient = 1 - entropy

    # 计算权重
    weight = difference_coefficient / np.sum(difference_coefficient)

    return weight


def topsis_score(data, weights):
    """
    使用TOPSIS方法计算综合得分
    :param data: 二维数组，每行代表一个样本，每列代表一个指标
    :param weights: 各指标对应的权重向量，长度需与指标数量一致
    :return: 各样本的TOPSIS综合得分
    """
    # 归一化数据
    normalized_data = data / np.sqrt(np.sum(data ** 2, axis=0))

    # 加权归一化数据
    weighted_data = normalized_data * weights

    # 确定正理想解和负理想解
    positive_ideal = np.max(weighted_data, axis=0)
    negative_ideal = np.min(weighted_data, axis=0)

    # 计算各样本与正理想解、负理想解的欧式距离
    distance_to_positive = np.sqrt(np.sum((weighted_data - positive_ideal) ** 2, axis=1))
    distance_to_negative = np.sqrt(np.sum((weighted_data - negative_ideal) ** 2, axis=1))

    # 计算相对贴近度（综合得分）
    score = distance_to_negative / (distance_to_positive + distance_to_negative)

    return score


def read_data(file_path):
    """
    从Excel文件读取数据，假设每个sheet对应一个项目，格式如描述中所示
    :param file_path: Excel文件路径
    :return: 以字典形式返回数据，键为项目名，值为对应的DataFrame（去除第一列项目名后的数值数据）
    """
    data_dict = {}
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        project_name = df.iloc[0, 0]
        data = df.iloc[:, 1:].values  # 去除第一列项目名
        data_dict[project_name] = data
    return data_dict

def sliding_window_entropy_topsis(data, window_size=2):
    """
    通过滑动窗口处理数据，并利用熵权法和TOPSIS进行指标合成
    :param data: 项目对应的数据矩阵（形状为 (样本数, 指标数 * 年数)）
    :param window_size: 滑动窗口大小，默认2年
    :return: 合成后的大指标数据
    """
    num_samples, num_cols = data.shape
    num_indicators = num_cols // 7  # 假设每7列对应一个小指标的数据
    new_data = []
    for i in range(num_samples):
        sample_data = data[i].reshape(num_indicators, 7)
        for j in range(2, 7):  # 从第3年开始才有完整窗口预测后续
            window_data = sample_data[:, j - window_size:j].reshape(1, -1)
            # 这里省略熵权法和TOPSIS具体实现步骤，只是示意后续添加的位置
            # 熵权法确定权重（假设已有函数实现entropy_weight）
            weights = entropy_weight(window_data)
            # TOPSIS计算综合得分（假设已有函数实现topsis_score）
            score = topsis_score(window_data, weights)
            new_data.append(score)
    return np.array(new_data).reshape(-1, 1)

def least_squares_fitting(data, target):
    """
    使用最小二乘法拟合数据
    :param data: 特征数据（大指标数据等）
    :param target: 目标数据（对应要预测的结果等）
    :return: 拟合的模型对象
    """
    model = LinearRegression()
    model.fit(data, target)
    return model


def predict_result(data_dict):
    """
    整体的预测流程，从数据读取到最终结果输出
    :param data_dict: 从Excel读取的原始数据字典（项目名 - 对应数据）
    :return: 预测结果字典（项目名 - 最终预测值（0或1））
    """
    results = {}
    for project_name, project_data in data_dict.items():
        processed_data = sliding_window_entropy_topsis(project_data)

        # 划分训练集和测试集（这里简单按比例划分，实际可调整）
        train_size = int(len(processed_data) * 0.8)
        train_data = processed_data[:train_size]
        test_data = processed_data[train_size:]

        # 中间层4个节点拟合（假设有4个目标值要先拟合预测）
        intermediate_targets = np.zeros((len(train_data), 4))
        for node in range(4):
            intermediate_model = least_squares_fitting(train_data, intermediate_targets[:, node].reshape(-1, 1))
            intermediate_preds = intermediate_model.predict(test_data)

        # 利用中间层预测结果拟合最终输出（预测是否包含项目，0 - 1）
        final_model = least_squares_fitting(intermediate_preds, np.array([0 if i < 0.5 else 1 for i in np.random.rand(len(intermediate_preds))]).reshape(-1, 1))
        final_pred = final_model.predict(intermediate_preds)
        results[project_name] = [0 if p < 0.5 else 1 for p in final_pred][0]
    return results

if __name__ == "__main__":
    file_path = "your_file_path.xlsx"  # 替换为实际的Excel文件路径
    data_dict = read_data(file_path)
    prediction_results = predict_result(data_dict)
    for project, result in prediction_results.items():
        print(f"项目 {project} 的预测结果为: {'包含' if result == 1 else '不包含'}")
