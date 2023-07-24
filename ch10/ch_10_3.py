# @Author   : Chow

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer as FA
from scipy.stats import zscore

filename = 'data10_3.csv'
data = pd.read_csv(filename, header=None)

db = zscore(data, ddof=1)  # 数据标准化
r = np.corrcoef(db.T)  # 相关系数矩阵

val,vec = np.linalg.eig(r)
cs = np.cumsum(val)
rate = val / cs[-1]  # 求贡献率
srate = sorted(rate, reverse=True)
ssrate = np.cumsum(srate)
print('特征值为:',val,'\n贡献率为:',srate,'\n累计贡献率为:',ssrate)

fa = FA(3, rotation='varimax')  # 构建模型 
fa.fit(db)
A = fa.loadings_  # 提取因子载荷矩阵
gx = np.sum(A ** 2, axis=0)  # 计算信息贡献
s2 = 1- np.sum(A ** 2, axis=1)  # 计算特殊方差
ss = np.linalg.inv(np.diag(s2))
f = ss @ A @ np.linalg.inv(A.T @ ss @ A)  # 计算因子得分函数系数
df = db @ f # 计算因子得分
pj = df @ gx / sum(gx)

# 输出
print('载荷矩阵:',np.round(A, 4))
print('特殊方差:',np.round(s2, 4))
print('各因子方差贡献:',np.round(gx, 4))
print('评价值:',np.round(pj, 4))

# 排名
l = len(pj)
ind0 = np.argsort(-pj)
ind = np.zeros(l)
ind[ind0] = np.arange(0, l)
print('排名:',ind + 1)