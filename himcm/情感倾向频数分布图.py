
import pandas as pd  
import matplotlib.pyplot as plt  
from matplotlib import font_manager  
  
# 设置Matplotlib配置参数  
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体  
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号  
  
# 你的代码...
  
df=pd.read_excel(r'C:\Users\wxhzyx020222\Desktop\市调\小红书_评论_正文情感分析_20240302175156223.xlsx')
  
# 使用pandas的value_counts方法计算score的频数  
score_counts = df['score'].value_counts().sort_index()  
  
# 计算总频数用于归一化  
total_counts = score_counts.sum()  
  
# 归一化频数得到概率  
probability_density = score_counts / total_counts  

# 设置Matplotlib配置参数以移除边框和纵轴  
plt.rcParams['axes.edgecolor'] = 'none'  # 设置边框颜色为白色（或者'none'以完全移除边框）  
plt.rcParams['axes.spines.right'] = False  
plt.rcParams['axes.spines.top'] = False  

# 绘制散点图  
plt.figure(figsize=(10, 6))  
plt.scatter(score_counts.index, probability_density, s=50, color='orange')  # s参数控制点的大小  



# 设置图表标题和坐标轴标签  
plt.title('情感得分分布情况')  
plt.xlabel('分数')  
plt.ylabel(' ')  
plt.gca().set_ylabel(' ')

  
# 设置x轴刻度标签，确保它们不会重叠  
plt.xticks(rotation=45)  
  
# 在score=0处添加一条带有箭头的纵轴线  
# 定义箭头的起点和终点  
arrow_start = (0, min(probability_density) - 0.025)  # 箭头的起点位置，稍微低于最小概率值  
arrow_end = (0, max(probability_density) + 0.025)  # 箭头的终点位置，稍微高于最大概率值  
  
# 绘制箭头  
plt.annotate('', xytext=arrow_start, xy=arrow_end,  
             arrowprops=dict(arrowstyle='->', linestyle='--', facecolor='orange', linewidth=1))  
  
# 显示网格  
plt.grid(True)  
  
# 显示图表  
plt.show()