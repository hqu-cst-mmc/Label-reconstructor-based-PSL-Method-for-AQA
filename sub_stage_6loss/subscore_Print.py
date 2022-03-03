import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import xlrd
import csv

### convert XLSX to CSV
# data_xls = pd.read_excel('./data_files/sub-value-for-print.xlsx')
# data_xls.to_csv('./data_files/1.csv',encoding='utf-8')


# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体

df = pd.read_excel('./data_files/sub-value-for-print.xlsx')  # 文件有中文,encoding= 'gbk'
fig, ax = plt.subplots(figsize=(80,5))
# axis = df.pivot("stage","num of samples","subvalues")
sns.heatmap(df, vmax=0.8, vmin=0.45, xticklabels=5, yticklabels = ['Start','Take-off','Dropping','Entry','Ending'],fmt='d',square=True,cmap='RdBu_r',center=0.6,cbar_kws={"orientation":"horizontal"
    ,"shrink":0.2,"pad":10})
# ax.set_title('feature heatmap', fontsize=5)

plt.xlabel('Num of Testing Samples', fontdict={'family':'Times New Roman','size':15},color='k')
plt.ylabel('Diving Stages',fontdict={'family':'Times New Roman','size':15},color='k')
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
plt.show()
# xticklabels=True,
#             yticklabels = True,