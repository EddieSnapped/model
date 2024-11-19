import json

# 输入 JSON 文件路径
input_file = "/Users/stone/pythons/wddl_what/himcm/normalized_tfidf_scores.json"
output_file = "tf-idf_output.txt"

# 读取 JSON 数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 将 JSON 转换为两列内容并写入输出文件
with open(output_file, "w", encoding="utf-8") as f:
    for key, value in data.items():
        f.write(f"{key}\t{value}\n")

print(f"转换完成！两列内容已写入 {output_file}")
