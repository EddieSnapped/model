import pandas as pd
import re

def process_excel(file_path, output_path):
    """
    处理 Excel 文件：
    1. 向下填充合并单元格值。
    2. 提取第一列中的年份和第二列中逗号前的值。
    3. 统计每一年每项的出现次数。
    4. 输出的行是不同项，列是递增年份。
    :param file_path: 输入 Excel 文件路径。
    :param output_path: 输出 Excel 文件路径。
    """
    # 读取数据并填充合并单元格
    data = pd.read_excel(file_path, header=None)
    data_filled = data.fillna(method='ffill')

    # 提取年份和项目
    data_filled['Year'] = data_filled[0].apply(lambda x: re.search(r'\b(19\d{2}|20\d{2})\b', str(x)).group(0) if re.search(r'\b(19\d{2}|20\d{2})\b', str(x)) else None)
    data_filled['Item'] = data_filled[1].apply(lambda x: str(x).split(',')[0] if pd.notna(x) else None)

    # 丢弃无效行（没有年份或项目的行）
    data_cleaned = data_filled.dropna(subset=['Year', 'Item'])

    # 统计每一年每项的出现次数
    stats = data_cleaned.groupby(['Year', 'Item']).size().unstack(fill_value=0)

    # 确保列按年份排序
    stats = stats.T.sort_index(axis=1)

    # 保存结果到 Excel 文件
    stats.to_excel(output_path)
    print(f"处理完成，结果保存到 {output_path}")

# 示例使用
file_path = "/Users/stone/Downloads/olympic_data/sanc.xlsx"  # 输入文件路径
output_path = "sanc_filtered.xlsx"  # 输出文件路径
process_excel(file_path, output_path)
