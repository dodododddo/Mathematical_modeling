import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from scipy.optimize import leastsq

def residuals(args, x, y): # 记得参数放前面
    a, b = args
    y_hat = b * x + a
    return y - y_hat

data = pd.read_csv('ch_8_6.csv')
data.drop('year',axis=1,inplace=True)
data_series = np.asarray(data).reshape((1,-1))[0]
time = np.arange(len(data_series))
data_log = np.log(data_series)

start_args = [0.5, 1]
arg = leastsq(residuals, start_args, args=(time, data_log))[0]
a, b = arg
data_hat = np.exp(b * time + a)

data_res = data_series - data_hat
with open('model.out','a+') as f:
    sys.stdout = f
    print('######################################################################################################')
    print('序列主成分预测情况:')
    print(f'square_loss = {np.sum(data_res ** 2):.4f}')
    print(f'avg_loss_rate = {100 * np.sqrt(np.mean(data_res ** 2)) / np.mean(data_series):.2f}%\n')
    
    print('arima模型选阶及其AIC:')
    model = auto_arima(data_res, max_p=50, max_q=50,max_d=5,d=2,start_p=5,start_q=5, start_P=5, start_Q=5, max_P=50,max_Q=50,seasonal=True,m=4,trace=True, information_criterion='aic')
    forecast_res = model.predict(8)
    forecast_main = np.exp(b * np.arange(100,108) + a)
    forecast = forecast_res + forecast_main

    print(f'\n预测值:{forecast}\n')

plt.plot(np.arange(0, 108), np.concatenate([data_series,forecast],axis=0), label=u'forecast')
plt.plot(time, data_series, label=u'res')
plt.show()



