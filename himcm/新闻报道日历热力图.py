
import pandas as pd
import calplot
import matplotlib.pylab as plt

# 读取数据
df = pd.read_excel(r'C:\Users\wxhzyx020222\Desktop\市调\新闻时间.xlsx')

# 统计每天的新闻数量
counts = df.groupby('date')['num'].agg('count').reset_index()

# 数据格式转换
counts['date'] = pd.to_datetime(counts['date'])

# 将订单时间设置为索引
counts.set_index('date', inplace = True)

# 绘制图形
pl2 = calplot.calplot(counts['num'], cmap = 'YlOrRd', textformat  ='{:.0f}', figsize = (16, 8), suptitle = "Total Orders by Month and Year")
plt.show()
