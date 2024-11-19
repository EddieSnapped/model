import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.utils import simple_preprocess
from torch.utils.data import Dataset, DataLoader
# from nltk.tokenize import word_tokenize
import numpy as np

with open('/Users/stone/pythons/wddl_what/himcm/stopword.txt', 'r', encoding='utf-8') as file:
    stop_words = set(line.strip() for line in file)
exclude_words = ['reply', 'hr', 'ago', 'hr . ago', 'more', 'replies', 'people', 'country', 'time', 'days', 'leg', 'title', 'deleted', 'edited']
stop_words.update(word.lower() for word in exclude_words)

dir_path = [
    # '/Users/stone/Downloads/olympic_data/Olympic_media_res.csv',
    # '/Users/stone/Downloads/olympic_data/reddit_olympic(0).csv',
    # '/Users/stone/Downloads/olympic_data/reddit_olympic(1).csv',
    # '/Users/stone/Downloads/olympic_data/reddit_olympic(2).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(3).csv',
    '/Users/stone/Downloads/olympic_data/reddit_olympic(4).csv',
    '/Users/stone/Downloads/olympic_data/olympic_report.csv'
]
topic_num = 3  # 假设我们想要的主题数量

def load_documents(file_path):
    df = pd.read_csv(file_path, header=None)  # 假设没有表头
    df.dropna(how='all', inplace=True)
    documents = [row.astype(str).values for _, row in df.iterrows()]
    return documents

def preprocess(text):
    return [word for word in simple_preprocess(text) if word not in stop_words]

def perform_BTM():
    corpus = []
    for i in dir_path:
        print("reading ", i)
        corpus.extend(load_documents(i))
    print("input done.")
    corpus = [preprocess(i) for i in corpus]
    print(len(corpus))


    # 构建词汇表
    vocab = set()
    for doc in corpus:
        vocab.update(doc)
    # vocab = list(vocab)
    word2id = {word: i for i, word in enumerate(vocab)}
    id2word = {i: word for i, word in enumerate(vocab)}

    # 构建biterm矩阵
    biterm_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.int32)
    for doc in corpus:
        doc_ids = [word2id[word] for word in doc]
        for i in range(len(doc_ids)):
            for j in range(i+1, len(doc_ids)):
                biterm_matrix[doc_ids[i], doc_ids[j]] += 1

    # 定义BTM模型
    class BTM(nn.Module):
        def __init__(self, vocab_size, topic_num):
            super(BTM, self).__init__()
            self.E = nn.Embedding(vocab_size, topic_num)
            self.topic_num = topic_num

        def forward(self, x):
            return torch.matmul(self.E(x), self.E.weight.t())

    # 模型参数
    vocab_size = len(vocab)

    # 实例化模型
    model = BTM(vocab_size, topic_num)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    def train(model, biterm_matrix, epochs=1000):
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(torch.from_numpy(np.arange(vocab_size)))
            loss = torch.sum((output - torch.from_numpy(biterm_matrix)) ** 2)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print('start training...')
    train(model, biterm_matrix)

    # 输出每个主题的关键词
    def get_topic_keywords(model: BTM, top_n=10):
        with torch.no_grad():
            # 获取词嵌入权重
            weight = model.E.weight
            # 计算每个词在每个主题中的得分
            # scores = torch.matmul(weight.t(), weight)
            scores = weight.t()
            # 获取每个主题的关键词索引
            top_words = torch.topk(scores, top_n, dim=1)[1]
            return top_words

    # 获取并打印每个主题的关键词
    topic_keywords = get_topic_keywords(model)
    for topic_idx, topic_words in enumerate(topic_keywords):
        print(f"Topic {topic_idx}:", end=' ')
        for word_idx in topic_words:
            print(f"{id2word[word_idx.item()]}", end=' ')
        print()


if __name__ == "__main__":
    perform_BTM()