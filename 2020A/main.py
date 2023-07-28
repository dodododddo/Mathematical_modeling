from utils import *
from model import Model


alphas=[6.80e-04, 6.58e-04, 7.74e-04, 5.09e-04, 4.76e-04]
temps = [175, 195, 235, 255, 25]
v = 70 / 60

model = Model(v, temps, alphas)
predicts = model.solve()

trans = pos2temp(temps)
positions = np.arange(0, 435.5, model.v * model.dt)
am_temps = trans(positions)

true_data = get_true_temp()
true_data[:,0] = true_data[:, 0] * model.v + 25
    
    
plt.xlabel('time/s')
plt.ylabel('T/℃')
plt.plot(positions, am_temps, label=u'ambient_temperature')
plt.plot(true_data[:,0], true_data[:, 1], label=u'true_temperature')
plt.plot(positions, predicts, label=u'predict_temperature')
plt.legend()
plt.show()