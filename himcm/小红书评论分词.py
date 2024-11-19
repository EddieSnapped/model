import jieba  
import re  
import csv  
  
# 创建停用词列表  
def stopwordslist():  
    stopwords = [line.strip() for line in open(r'C:\Users\wxhzyx020222\Desktop\市调\stopwords.txt', encoding='utf-8').readlines()]  
    return set(stopwords)  # 使用集合来加速查找  
  
# 数据清洗函数  
def processing(text):  
    text = re.sub("@.+?( |$)", "", text)           # 去除 @xxx (用户名)  
    text = re.sub("【.+?】", "", text)             # 去除 【xx】 (里面的内容通常都不是用户自己写的)  
    text = re.sub(".*?:", "", text)                # 去除微博用户的名字  
    text = re.sub("#.*#", "", text)                # 去除话题引用  
    text = re.sub("\n", "", text)  
    return text.strip()  # 确保返回的是没有前后空格的字符串  
  
# 分词函数  
def seg_depart(sentence, stopwords):  
    jieba.load_userdict(r'C:\Users\wxhzyx020222\Desktop\市调\保留词.txt')  
    sentence_depart = jieba.cut(sentence)  
    outstr = ' '.join([word for word in sentence_depart if word not in stopwords and word != '\t'])  
    return outstr  
  
# 给出文档路径    
input_filename = r'C:\Users\wxhzyx020222\Desktop\市调\小红书_评论.csv'  # 原文档路径    
output_filename = r'C:\Users\wxhzyx020222\Desktop\市调\output.csv'  # 输出文档路径    
  
# 使用 csv.reader 读取 CSV 文件    
with open(input_filename, 'r', encoding='utf-8-sig') as csvfile:  
    reader = csv.reader(csvfile, delimiter=',', quotechar='"', doublequote=False)  
      
    # 使用 csv.writer 写入 CSV 文件  
    with open(output_filename, 'w', newline='', encoding='utf-8-sig') as output_csvfile:  
        writer = csv.writer(output_csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  
          
        # 遍历 CSV 文件的每一行  
        for line in reader:  
            # 假设我们只需要处理第一列  
            text = line[0]  
              
            # 对文本进行清洗  
            cleaned_text = processing(text)  
              
            # 对清洗后的文本进行分词  
            stopwords = stopwordslist()  
            segmented_text = seg_depart(cleaned_text, stopwords)  
              
            # 写入分词后的文本到新的 CSV 文件  
            writer.writerow([segmented_text])  
  
print("分词成功！！！")