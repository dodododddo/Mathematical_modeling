from utils import * 
from scipy import linalg
from math import floor, ceil

pos_anchors = [0, 25, 197.5, 339.5, 410.5, 435.5]

class Model():
    def __init__(self, v: float, temps: list, alphas: list, dx: float = 0.0001, dt: float = 0.5, l = 0.015):
        # v单位为cm/s
        self.v = v
        self.pos2am_temp = pos2temp(temps)
        # 作为边界条件的步数
        self.step_anchors = [floor(pos_anchor / v / dt) + 1 for pos_anchor in pos_anchors]
        # 为了与后续步骤统一
        self.step_anchors[0] = 1
        self.T = floor(pos_anchors[-1] / v)
        # 时间总步数
        self.Nt = floor(self.T / dt) + 1
        self.Nx = floor(l / dx) + 1
        self.alphas = alphas
        self.rs = [(alpha ** 2) * dt / (dx ** 2) for alpha in alphas]
        self.dx = dx
        self.dt = dt
        self.l = l
    

    def solve(self):
        '''计算所有时间的T值'''
        u = np.zeros((self.Nx, self.Nt))
        u[:, 0] = 25
        res = np.ones(self.Nt) * 25
        t0 = np.arange(0, self.T + self.dt, self.dt)
        u0 = np.array([self.pos2am_temp(self.v * time) for time in t0])
        # 测温点
        test_pos = ceil(self.l / (2 * self.dx))

        for i in range(len(self.step_anchors) - 1):
            for j in range(self.step_anchors[i] - 1, self.step_anchors[i + 1] - 1):
                A = get_trans_matrix(self.rs[i], size=self.Nx)
                d = np.zeros(self.Nx)
                d[0] = self.rs[i] * u0[j + 1]
                d[-1] = d[0]
                u[:, j + 1] = linalg.solve(A, d + u[:, j])
                res[j + 1] = u[test_pos, j + 1]

        return res
            
    
class Optimizer():
    def __init__():
        pass

    def __impl(self, model:Model, *args):
        raise NotImplementedError()
    
    def optimize(self, model:Model, *args):
        self.__impl()


class LeastsqOptimizer(Optimizer):
    def __init__():
        super().__init__()
   
if __name__ == '__main__':

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
    
    