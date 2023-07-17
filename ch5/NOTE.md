# ch5: 插值与拟合

## 插值方法
* 线性插值
* 三次样条插值: 周期条件与自然条件

## 拟合方法
* 最小二乘
* 先对原始数据求对数/滑动平均再进行最小二乘拟合
* 实用工具: leastsq

```

def residuals(args, x, y): # 记得参数放前面
    a, b = args
    y_hat = b * x + a
    return y - y_hat

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([15.3, 20.5, 27.4, 36.6, 49.1, 65.6, 87.87, 117.6])

z = np.log(y)

arg_origin = [0.5, 1]

res = leastsq(residuals, arg_origin, args=(x, z))
a, b = res[0]
y_hat = np.exp(b * x + a)
loss = np.sum((y - y_hat) ** 2)
```