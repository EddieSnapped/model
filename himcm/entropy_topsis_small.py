import pandas as pd
import numpy as np

need_topsis = True


def handle_null_values(data):
    """
    处理缺失值（null）。可以选择填充或删除缺失值。
    :param data: 输入的数据矩阵
    :return: 处理后的数据矩阵
    """

    # 将 'null' 替换为 NaN (pandas 标准的缺失值标识符)
    data.replace("null", np.nan, inplace=True)

    # 可以选择填充 NaN，这里填充为该列的均值
    data = data.fillna(data.mean())

    # 也可以选择删除包含 NaN 的行
    # data = data.dropna(axis=0)  # 删除含有缺失值的行
    return data


def entropy_weight_method(data):
    """
    计算熵权法权重
    :param data: 指标矩阵，行是样本，列是指标
    :return: 每个指标的原始权重
    """
    # 标准化数据
    data_norm = data / data.sum(axis=0)

    # 计算信息熵
    epsilon = 1e-10  # 防止 log(0)
    entropy = -np.sum(data_norm * np.log(data_norm + epsilon), axis=0) / np.log(len(data))

    # 计算冗余度
    redundancy = 1 - entropy

    # 计算权重
    weights = redundancy / np.sum(redundancy)
    return weights


def topsis(matrix, weights):
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
    ideal_best = weighted_matrix.max(axis=0)
    ideal_worst = weighted_matrix.min(axis=0)

    # 计算每个项目与理想解/反理想解的距离
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    # 计算得分
    scores = dist_worst / (dist_best + dist_worst + 1e-9)
    return scores


def standarize(data):
    standarized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return standarized_data


# def normalize(weights):
#     """
#     对熵权法计算出的权重进行归一化
#     :param weights: 权重数组
#     :return: 归一化后的权重
#     """
#
#     weights = weights / np.sum(weights)
#     return normalized_weights


def process_excel_flatten_normalized(file_path, output_path):
    """
    处理多 sheet 的 Excel 文件，计算熵权法权重并归一化。
    :param file_path: 输入 Excel 文件路径
    :param output_path: 输出权重结果的 Excel 文件路径
    """
    # 打开 Excel 文件
    xls = pd.ExcelFile(file_path)
    combined_data = pd.DataFrame()
    sheet_weights = {}
    sheet_num = []
    sheet_sum = []

    # 遍历所有 sheet，将所有数据合并
    for sheet_name in xls.sheet_names:
        # 读取当前 sheet
        sheet_data = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

        # 处理缺失值（"null"）
        sheet_data = handle_null_values(sheet_data.iloc[:, 1:])

        # 初始化列索引
        cols = sheet_data.columns
        i = 1  # 跳过第一列（列名）

        # 遍历每个指标
        count = 0
        while i < len(cols):
            # 提取当前指标的 7 列数据
            indicator_name = cols[i][:-5]
            indicator_data = sheet_data.iloc[:, i:i + 7]

            # 如果是空列，跳过
            if indicator_data.isnull().all().all():
                i += 1
                continue

            # 将 7 列数据按行合并为单列
            flattened = indicator_data.values.flatten()
            flattened = standarize(flattened)
            flattened_df = pd.DataFrame(flattened, columns=[indicator_name])


            # 如果 combined_data 为空，初始化为第一个指标的数据
            if combined_data.shape[0] == 0:
                combined_data = flattened_df
            else:
                # 保证行数一致，若不一致，则裁剪至最小行数
                min_len = min(combined_data.shape[0], flattened_df.shape[0])
                combined_data = combined_data.iloc[:min_len]
                flattened_df = flattened_df.iloc[:min_len]
                combined_data[indicator_name] = flattened_df[indicator_name].values

            # 跳过空列和下一组指标
            i += 8  # 跳过当前指标后面的空列（8列）
            count += 1  # 记录sheet上的指标数
        sheet_num.append(count)


    # 对合并后的数据计算熵权
    raw_weights = entropy_weight_method(combined_data)

    if need_topsis:

        # 计算 TOPSIS 得分

        m = 0
        n = 3
        topsis_output = pd.DataFrame()
        topsis_output['1'] = topsis(combined_data.iloc[:, m:n], raw_weights.iloc[m:n])
        m += n
        n += 2
        topsis_output['2'] = topsis(combined_data.iloc[:, m:n], raw_weights.iloc[m:n])
        m += n
        n += 2
        topsis_output['3'] = topsis(combined_data.iloc[:, m:n], raw_weights.iloc[m:n])
        m += n
        n += 1
        topsis_output['4'] = topsis(combined_data.iloc[:, m:n], raw_weights.iloc[m:n])
        m += n
        n += 3
        topsis_output['5'] = topsis(combined_data.iloc[:, m:n], raw_weights.iloc[m:n])
        m += n
        n += 2
        topsis_output['6'] = topsis(combined_data.iloc[:, m:n], raw_weights.iloc[:n])


    # 打印合并数据的熵权
    print(raw_weights)
    print(f"\n[熵权法{'+topsis' if need_topsis else ''}计算得到的权重]:")
    if need_topsis:
        print(topsis_output)

    # 对每个 Sheet 单独进行归一化
    ct = 0
    for i in sheet_num:
        raw_weights.iloc[ct:ct + i] = raw_weights.iloc[ct:ct + i] / raw_weights[ct:ct + i].sum()
        ct += i

    print(f"\n归一化权重:")
    print(raw_weights)
    # 将所有 sheet 的归一化后的权重保存到 Excel 文件
    # result_df = pd.DataFrame(sheet_weights).T
    # result = pd.DataFrame({
    #     "指标名称": combined_data.keys(),
    #     "权重": raw_weights
    # })

    if need_topsis:
        output_path = "output_topsis.xlsx"
        topsis_output.to_excel(output_path, index=True)
    else:
        raw_weights.to_excel(output_path, index=True)
    print(f"处理完成，权重结果保存到 {output_path}")


if __name__ == "__main__":
    file_path = "/Users/stone/Downloads/olympic_data/train_data.xlsx"  # 输入文件路径
    output_path = "output_weights_small.xlsx"  # 输出文件路径
    process_excel_flatten_normalized(file_path, output_path)
