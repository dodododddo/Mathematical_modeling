import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy import linalg
from scipy.optimize import leastsq

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
print(y - y_hat)

pl.plot(x, y, marker='+', label=u"real_data")
pl.plot(x, y_hat, marker='*', label=u"fitting_data")
pl.legend()
pl.show()