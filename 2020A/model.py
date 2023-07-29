from utils import * 
from scipy import linalg
from math import floor, ceil, cos
from copy import deepcopy

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
        # self.rs = [(alpha ** 2) * dt / (dx ** 2) for alpha in alphas]
        self.dx = dx
        self.dt = dt
        self.l = l
    

    def solve(self):
        '''计算所有时间的T值'''
        rs = [(alpha ** 2) * self.dt / (self.dx ** 2) for alpha in self.alphas]
        u = np.zeros((self.Nx, self.Nt))
        u[:, 0] = 25
        res = np.ones(self.Nt) * 25
        t0 = np.arange(0, self.T + self.dt, self.dt)
        u0 = np.array([self.pos2am_temp(self.v * time) for time in t0])
        # 测温点
        test_pos = ceil(self.l / (2 * self.dx))

        for i in range(len(self.step_anchors) - 1):
            for j in range(self.step_anchors[i] - 1, self.step_anchors[i + 1] - 1):
                A = get_trans_matrix(rs[i], size=self.Nx)
                d = np.zeros(self.Nx)
                d[0] = rs[i] * u0[j + 1]
                d[-1] = d[0]
                u[:, j + 1] = linalg.solve(A, d + u[:, j])
                res[j + 1] = u[test_pos, j + 1]

        return res
    
    def optimize_each_epoch(self, lr):
        positions = np.arange(0, 435.5, model.v * model.dt)
        am_temps = self.pos2am_temp(positions)
        true_data = get_true_temp()[:, 1]
        pred = self.solve()
        for i in range(len(self.step_anchors) - 1):
    
            if (pred[self.step_anchors[i] + 38: self.step_anchors[i + 1]] - true_data[self.step_anchors[i]: self.step_anchors[i + 1] - 38]).sum() >= 0:
                if(true_data[self.step_anchors[i]] < am_temps[self.step_anchors[i] + 38]):
                    self.alphas[i] -= lr
                else:
                    self.alphas[i] += lr
            if (pred[self.step_anchors[i] + 38: self.step_anchors[i + 1]] - true_data[self.step_anchors[i]: self.step_anchors[i + 1] - 38]).sum() < 0:
                if(true_data[self.step_anchors[i]] > am_temps[self.step_anchors[i] + 38]):
                    self.alphas[i] -= lr
                else:
                    self.alphas[i] += lr

    def optimize(self, lr):
        pass

    
            
class Optimizer():
    def __init__(self, model:Model, lr):
        self.lr = lr
        self.model = model
        self.true_data = get_true_temp()[:,1]

    def optimize(self, num_epoches: int):
        for epoch in range(num_epoches):
            lr = self.lr * cos(3.1415926 / 3 * epoch / num_epoches)
            pred = self.model.solve()[38:]
            l = loss(self.true_data, pred)
            delta = [self.calculate_grad(l, i) * lr for i in range(5)]
            self.model.alphas = [self.model.alphas[i] - delta[i] for i in range(5)]
            print(f'epoch{epoch + 1} done')
        
    def calculate_grad(self, loss_origin, idx):
        model = deepcopy(self.model)
        step = 1e-7
        model.alphas[idx] += step
        pred_new = model.solve()[38:]
        # print(f'原模型: alpha[i]={self.model.alphas[idx]}, 计算模型: alpha[i] = {model.alphas[idx]}')
        loss_new = loss(self.true_data, pred_new)
        grad = (loss_new - loss_origin) / step
        return grad


class LeastsqOptimizer(Optimizer):
    def __init__():
        super().__init__()

def loss(y, y_hat):
    return ((y - y_hat) ** 2).sum()




   
if __name__ == '__main__':

    alphas = [0.0006293750934271579, 0.0006827143001546297, 0.0007873334421703878, 0.0005134110423806804, 0.0004619238287142745] 
    temps = [175, 195, 235, 255, 25]
    v = 70 / 60

    model = Model(v, temps, alphas)
    lr = 1e-13
    num_epoches = 15
    optimizer = Optimizer(model, lr)
    optimizer.optimize(num_epoches)

    predicts = model.solve()

    trans = pos2temp(temps)
    positions = np.arange(0, 435.5, model.v * model.dt)
    am_temps = trans(positions)

    true_data = get_true_temp()
    true_data[:,0] = true_data[:, 0] * model.v + 25
    
    print(f'loss = {loss(true_data[:, 1], predicts[38:])}')
    print(f'alphas = {model.alphas}')
    plt.xlabel('time/s')
    plt.ylabel('T/℃')
    plt.plot(positions, am_temps, label=u'ambient_temperature')
    plt.plot(true_data[:,0], true_data[:, 1], label=u'true_temperature')
    plt.plot(positions, predicts, label=u'predict_temperature')
    plt.legend()
    plt.show()
    
    