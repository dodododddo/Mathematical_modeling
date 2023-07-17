# 时序数据处理
## 序列性质
1.严平稳性: 分布不随时间变化
2.宽平稳性: 一阶矩、二阶矩不随时间变化
3.自相关性: 序列数据同n期滞后序列数据的相关性
4.偏自相关性: 序列数据与n期滞后序列数据剔除n期滞后数据之间的数据的相关性

## 检验方式
1.ADF检验: 得到ADF值与P值，用于判断序列平稳性
2.ACF检验: 用于查看序列自相关性来选择PQ值
3.PACF检验: 用于查看序列偏自相关性来选择PQ值

## 常用模型:ARIMA
超参:
* p: AR模型往前看几个数据
* q: MA模型往前看几个数据
* d: 差分次数

## 常用预处理方式
1. 先用线性模型/高次函数/指数模型/对数模型拟合大趋势，可以先取周期内平均数来拟合，再处理残差序列
2. 差分，包括周期差分，多次差分来获得平稳数据
3. 滑动平均、取对数
4. 若序列数据有多个维度且希望融合多个维度数据进行预测，则要先对每个维度进行归一化

## 常用工具
* np.diff
* pm.auto_arima