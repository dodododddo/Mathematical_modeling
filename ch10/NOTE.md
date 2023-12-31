# 多元分析方法: 聚类与降维

## 聚类
### 样本聚类
#### 常用样本间相似性度量:
计算距离前先将数据进行标准化，消除不同feature之间的规模差异
1. 切比雪夫距离: 对应L无穷范数， 关注差距最大的维度
2. 欧几里德距离: 即日常语境中的距离
3. 马氏距离: 在欧几里德距离的基础上加入了协方差矩阵(通常由样本协方差矩阵估计),确保对线性变换的不变性,避免变量的多重相关性
4. 其他: 夹角余弦等

#### 常用变量间相似性度量:
* 相关系数矩阵
* 协方差矩阵

#### 常用类间相似性度量:
1. 最短、最长距离法
2. 重心法
3. 两两计算距离后取平均

#### 聚类算法
* 划分式: k-means(效率较好)
* 密度方法: OPTICS
* 层次化方法: Aggromerative(效率较慢，但可控制误差传递，得到不同相似水平下的聚类)


## 降维(1): 主成分分析
基本原理为用原变量的线性组合构造正交的新变量并使该变量的方差尽可能大
以下为一些注意点:
* 必须保证各变量本身有较高的相关性
* 计算前要先将数据标准化
* 选取主成分时不仅要看贡献率之和是否已满足要求，也要确保不存在原特征与现有主成分正交
* 如果用于作回归分析，则特征向量正负号无需关注；若直接用贡献率来作为主成分的系数，则必须作解释

## 降维(2): 因子分析
与主成分相反，因子分析试图从多个特征中分离出少量的共有因子，并用这些因子加上随机因素来线性表达原有特征。优点是可解释性较强。

计算流程为:
1.确认变量之间有较高相关性
2.计算因子载荷矩阵，并验证$R-\Lambda \Lambda^T$(特殊因子的方差)是否足够小
3.为提高因子的可解释性，可以对因子进行旋转，使得每个因子在不同变量上的载荷的方差/四次方值最大。$F \Rightarrow T^TF$ & $\Lambda \Rightarrow \Lambda T$
4.估计因子得分矩阵，赋予因子测度从而方便后续使用。主要方法包括加权最小二乘(最小化特殊因子方差)和回归方法，后者表达式为$\hat{F} = X_0R^{-1}\Lambda$
