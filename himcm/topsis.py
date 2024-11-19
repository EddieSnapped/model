import numpy as np
import pandas as pd

file_path = "/Users/stone/Downloads/olympic_data/indicators.xlsx"  # 替换为实际文件路径


def handle_missing_data(matrix, strategy="mean"):
    """
    缺失值处理
    :param matrix: 输入矩阵，可能包含 NaN (numpy.ndarray)
    :param strategy: 填充策略 ("mean", "median", "zero")
    :return: 填充后的矩阵
    """
    # 将 0 替换为 NaN，便于统一处理
    matrix = np.where(matrix == 0, np.nan, matrix)

    if strategy == "mean":
        fill_values = np.nanmean(matrix, axis=0)  # 按列计算均值
    elif strategy == "median":
        fill_values = np.nanmedian(matrix, axis=0)  # 按列计算中位数
    elif strategy == "zero":
        fill_values = np.zeros(matrix.shape[1])  # 用 0 填充
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'zero'.")

    # 填充缺失值
    for i in range(matrix.shape[1]):
        matrix[np.isnan(matrix[:, i]), i] = fill_values[i]

    return matrix


def entropy_weight_method(matrix):
    """
    熵权法计算指标权重
    :param matrix: 输入矩阵 (numpy.ndarray)，形状为 (m, n)
    :return: 每个指标的权重
    """
    # 数据标准化：归一化为 [0,1] 区间
    min_vals = matrix.min(axis=0)
    max_vals = matrix.max(axis=0)
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals + 1e-9)

    # 计算比例值
    row_sum = normalized_matrix.sum(axis=0)
    proportion = normalized_matrix / (row_sum + 1e-9)

    # 计算熵值
    entropy = -np.nansum(proportion * np.log(proportion + 1e-9), axis=0) / np.log(matrix.shape[0])

    # 计算权重
    redundancy = 1 - entropy  # 冗余度
    weights = redundancy / redundancy.sum()

    return weights


def topsis(matrix, weights, is_benefit):
    """
    TOPSIS 方法计算项目得分
    :param matrix: 输入矩阵 (numpy.ndarray)，形状为 (m, n)
    :param weights: 指标权重 (numpy.ndarray)，形状为 (n,)
    :param is_benefit: 指标类型 (list)，True 表示收益型，False 表示成本型
    :return: 每个项目的得分
    """
    # 数据标准化
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

    # 加权标准化
    weighted_matrix = norm_matrix * weights

    # 确定理想解和反理想解
    ideal_best = np.where(is_benefit, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_worst = np.where(is_benefit, weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))

    # 计算每个项目与理想解/反理想解的距离
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    # 计算得分
    scores = dist_worst / (dist_best + dist_worst + 1e-9)
    return scores


# 从 Excel 中读取数据
data = pd.read_excel(file_path)

# 提取指标数据（跳过第一列和第一行）

indicators = data.keys()[1:]  # 指标名称
projects = data.iloc[:, 0].values  # 项目名称
matrix = data.iloc[:, 1:].values.astype(float)

# 处理缺失值（按均值填充）
matrix = handle_missing_data(matrix, strategy="mean")

# 计算权重（熵权法）
weights = entropy_weight_method(matrix)

# 定义指标类型（True 表示收益型，False 表示成本型）
is_benefit = [True] * matrix.shape[1]  # 假设所有指标为收益型

# 计算 TOPSIS 得分
scores = topsis(matrix, weights, is_benefit)

# 输出结果
results = pd.DataFrame({
    "项目": projects,
    "TOPSIS 得分": scores
})
for i, weight in enumerate(weights):
    print(f"指标 {indicators[i]} 的权重: {weight:.4f}")
print("\nTOPSIS 得分:")
print(results)
