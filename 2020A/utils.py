from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import time

def get_true_temp(path: str = 'data.xlsx'):
    data = pd.read_excel(path).to_numpy()
    # 从进入区域开始计时
    data[:,0] -= 19
    return data


def pos2temp(temps:list = [175, 195, 235, 255, 25])->callable:
    
    '''接受五个大温区的温度列表，返回一个插值函数'''

    assert(len(temps) == 5)
    # 锚点温度，即每个大温区左右两端的温度
    anchor_positions = [0, 25, 197.5, 202.5, 233, 238, 268.5, 273.5, 339.5, 344.5, 410.5, 435.5]
    anchor_temps = [None] * len(anchor_positions)
    # 炉前区域(室温)
    anchor_temps[0] = 25
    # 大温区1: 小温区1-5
    anchor_temps[1] = temps[0]
    anchor_temps[2] = temps[0]
    # 大温区2: 小温区6
    anchor_temps[3] = temps[1]
    anchor_temps[4] = temps[1]
    # 大温区3: 小温区7
    anchor_temps[5] = temps[2]
    anchor_temps[6] = temps[2]
    # 大温区4: 小温区8-9
    anchor_temps[7] = temps[3]
    anchor_temps[8] = temps[3]
    # 大温区5: 小温区10-11
    anchor_temps[9] = temps[4]
    anchor_temps[10] = temps[4]
    # 炉后区域(室温)
    anchor_temps[11] = 25
    func = interp1d(anchor_positions, anchor_temps)
    return func

def get_trans_matrix(r: float, size: int):
    ''' 传入r, 计算三对角矩阵 '''
    mat1 = np.eye(size) * (1 + 2 * r)
    mat2 = np.eye(size, k=1) * -r
    mat3 = np.eye(size, k=-1) * -r
    return mat1 + mat2 + mat3
    
class Timer():
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist


if __name__ == "__main__":

    trans = pos2temp()
    test_temp = trans(200)
    assert(test_temp == 185)
    print(f'小温区5和小温区6之间区域中点温度为{test_temp}')
    print(get_trans_matrix(1, 4))
    print(get_true_temp())
    