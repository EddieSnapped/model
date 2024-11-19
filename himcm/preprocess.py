import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


dir_path = [
    '/Users/stone/Downloads/olympic_data/Olympic_media_res.csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(0).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(1).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(2).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(3).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(4).csv',
    '/Users/stone/Downloads/olympic_data/olympic_report.csv'
]


output_file = 'processed_output.txt'
exclude_words = ['reply', 'hr', 'ago', 'hr . ago', 'more', 'replies', 'people', 'country', 'time', 'days', 'leg', 'title', 'day']  # 要排除的词汇列表

# 下载 NLTK 所需的资源
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

def process_csv_files():
    """
    处理指定目录中的所有 CSV 文件，删除空行，提取第一列内容，拼接后分词，去除停用词和特定词汇，并将结果输出到文件。
    """
    combined_text = []

    # stop_words = set(stopwords.words('english'))
    # read stoplist, use set to improve efficiency
    with open('/Users/stone/pythons/wddl_what/himcm/stopword.txt', 'r', encoding='utf-8') as file:
        stop_words = set(line.strip() for line in file)

    # additional stop words
    stop_words.update(word.lower() for word in exclude_words)

    # loop though csv1~5
    # for abc in range(6):
    for filename in dir_path:
        # filename = dir_path(abc)
        # filename = '/Users/stone/Downloads/olympic_data/olympic_report.csv'
        if filename.endswith('.csv'):
            file_path = filename
            try:
                # read csv
                df = pd.read_csv(file_path, header=None)

                # delete black lines
                df.dropna(how='all', inplace=True)

                # extract contents
                first_column_text = df.astype(str).values.flatten().tolist()
                combined_text.extend(first_column_text)
                print(f"成功导入文件 {filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错：{e}")

    # concatenate
    full_text = ' '.join(combined_text)
    print("拼接")

    # tokenize
    words = word_tokenize(full_text)

    # filter stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # write results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(filtered_words))

    print(f"处理结果已保存到 {output_file}")


if __name__ == "__main__":
    process_csv_files()
