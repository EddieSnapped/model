import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

# nltk.download('stopwords')
with open('/Users/stone/pythons/wddl_what/himcm/stopword.txt', 'r', encoding='utf-8') as file:
    stop_words = set(line.strip() for line in file)
exclude_words = ['reply', 'hr', 'ago', 'hr . ago', 'more', 'replies', 'people', 'country', 'time', 'days', 'leg', 'title', 'deleted', 'edited']
stop_words.update(word.lower() for word in exclude_words)

dir_path = [
    '/Users/stone/Downloads/olympic_data/Olympic_media_res.csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(0).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(1).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(2).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(3).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(4).csv',
    '/Users/stone/Downloads/olympic_data/olympic_report.csv'
]
def load_documents(file_path):
    df = pd.read_csv(file_path, header=None)  # 假设没有表头
    # 将每行内容合并成一个完整的文档
    documents = [" ".join(row.dropna().astype(str).values) for _, row in df.iterrows()]
    return documents

def preprocess(text):
    return [word for word in simple_preprocess(text) if word not in stop_words]


# 主函数：执行 LDA 主题建模
def perform_lda(num_topics=2, passes=10):
    # 加载文档
    documents = []
    for i in dir_path:
        print("reading ", i)/
        documents.extend(load_documents(i))

    # 预处理文档
    processed_docs = [preprocess(doc) for doc in documents]

    # 创建词典和语料库
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # 训练 LDA 模型
    print("start training...")
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

    # 输出每个主题的关键词
    for idx, topic in lda_model.print_topics(-1):
        print(f"主题 {idx + 1}: {topic}")


# 示例：调用 LDA 模型
perform_lda(num_topics=2, passes=10)  # 设定主题数量和训练轮数
