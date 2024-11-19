import pandas as pd
import re


def filter_columns_by_year_with_prefix(file_path, output_path):
    """
    筛选 Excel 文件中第 0 行包含晚于 1992 且早于 2016 年的年份列（支持列名包含前缀），同时保留前两列。
    :param file_path: 输入 Excel 文件路径
    :param output_path: 输出 Excel 文件路径
    """
    # 读取 Excel 数据
    data = pd.read_excel(file_path, header=0)

    # 提取第 0 行（表头）
    headers = data.columns

    # 保留的列索引（包括前两列，无论是否满足年份条件）
    keep_columns = list(headers[:2])  # 前两列始终保留

    # 遍历表头中的其余列，筛选满足年份条件的列
    for col in headers[2:]:
        # 正则表达式匹配年份
        match = re.search(r'(19\d{2}|20\d{2})', str(col))  # 匹配列名中任何年份
        if match:
            year = int(match.group())  # 提取年份部分
            if 1992 <= year <= 2016:  # 判断是否符合条件
                keep_columns.append(col)  # 满足条件，保留列

    # 筛选 DataFrame
    filtered_data = data[keep_columns]

    # 保存筛选后的结果到新文件
    filtered_data.to_excel(output_path, index=False)
    print(f"筛选完成，结果已保存到 {output_path}")


# 示例使用
file_path = "/Users/stone/Downloads/olympic_data/sjgxbb.xlsx"  # 输入文件路径
output_path = "output_filtered.xlsx"  # 输出文件路径
filter_columns_by_year_with_prefix(file_path, output_path)
