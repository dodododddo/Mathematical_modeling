# @Author   : Chow

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import zscore

filename = 'data10_2.csv'
data = pd.read_csv(filename, header=None)

db = zscore(data, ddof=1)  # 数据标准化
md = PCA(n_components=None).fit(db)  # 构建模型

print('特征值为:',md.explained_variance_)
print('各主成分贡献率:',md.explained_variance_ratio_)
xs = md.components_
r1 = md.explained_variance_ratio_
print('主成分系数:\n',np.round(xs, 4))  # 系数保留4位小数
print('各主成分累计贡献率:',np.cumsum(md.explained_variance_ratio_))

n = 3
f = db@(xs[:n,:].T) # 矩阵乘法运算，得主成分得分
g = f@r1[:n]
print('主成分评价得分:',np.round(g, 4))

# 排序
l = len(g)
ind1 = np.argsort(-g)
ind11 = np.zeros(l)
ind11[ind1] = np.arange(0, l)
print(ind1+1984)
print('排名:',ind11+1)