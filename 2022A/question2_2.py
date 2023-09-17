from question1_2 import solve_all
import numpy as np


def ave_power(b, p):
    """定义目标函数"""
    x1 = np.array([-2])
    v1 = np.array([0])
    x2 = np.array([-1.8])
    v2 = np.array([0])
    x1, v1, x2, v2 = solve_all(x1, v1, x2, v2, m3=1165.992, omega=2.2143, f=4890, b0=b, p=p, b1=167.8395)
    y = b * (v1 - v2) ** 2
    all_power = 0
    for i in range(len(y) - 1):
        all_power += (y[i] + y[i+1]) * 0.2 / 2
    ave_power = all_power / ((len(y) -   1) * 0.2)
    return ave_power


DNA_SIZE = 24 # DNA长度 二进制编码长度
POP_SIZE = 200 # 初始种群数量
N_GENERATIONS = 50  # 进化代数

X_BOUND = [0.9, 1]
Y_BOUND = [0.9, 1]

def translateDNA(pop):
    '''解码'''
    x_pop = pop[:, 1::2] # 奇数列表示x
    y_pop = pop[:, ::2] # 偶数列表示y
    # pop:(POP_SIZE * DNA_SIZE) * (DNA_SIZE, 1) --> (POP_SIZE, 1) 完成解码
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y

def translateDNA_1(pop):
    '''解码'''
    x_pop = pop[1::2] # 奇数列表示x
    y_pop = pop[::2] # 偶数列表示y
    # pop:(1 * DNA_SIZE) * (DNA_SIZE, 1) --> (1, 1) 完成解码
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y

def get_fitness(pop):
    '''求最大值的适应度函数'''
    x, y = translateDNA(pop)
    pred = [0] * POP_SIZE
    for i in range(POP_SIZE):
        pred[i] = ave_power(x[i] * 1e5, y[i])
    return (pred - np.min(pred)) + 1e-3 # 防止适应度出现负值

def select(pop, fitness):
    '''自然选择, 适应度高的被选择机会多'''
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=(fitness)/(fitness.sum())) # 轮盘赌选择
    return pop[idx]

def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    '''交叉、变异'''
    new_pop = []
    for father in pop: # 遍历种群中的每一个个体，将该个体作为父亲
        child = father # 孩子先得到父亲的全部基因
        mother = pop[np.random.randint(POP_SIZE)] # 在种群中选择另一个个体，并将该个体作为母亲
        x_child, y_child = translateDNA_1(child)
        x_mother, y_mother = translateDNA_1(mother)
        if ave_power(x_child, y_child) < ave_power(x_mother, y_mother): # 基因型差
            if np.random.rand() < CROSSOVER_RATE: # 一定概率发生交叉
                cross_points = np.random.randint(low=0, high=DNA_SIZE * 2) # 随机产生交叉的点
                if cross_points % 2 == 0:
                    child[cross_points:] = mother[cross_points:] # 孩子得到位于交叉点后母亲的基因
                else:
                    child[:cross_points] = mother[:cross_points] # 孩子得到位于交叉点前母亲的基因
            else:
                mutation(child)  # 基因型差后代有一定的机率发生变异
        new_pop.append(child)
    return new_pop


def mutation(child, MUTATION_RATE=0.8):
    '''突变'''
    if np.random.rand() < MUTATION_RATE: # 以 MUTATION_RATE 的概率进行变异
        mutate_point = np.random.choice(np.arange(0, DNA_SIZE * 2), size=5, replace=False)
        child[mutate_point] = child[mutate_point] ^ 1 # 将变异点进行二进制反转

def print_info(pop):
    '''打印基因型'''
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    x, y = translateDNA(pop)
    print('最优的基因型:', pop[max_fitness_index])
    print('(x, y):', (x[max_fitness_index], y[max_fitness_index]))
    print('此时最优解:', ave_power(x[max_fitness_index] * 1e5, y[max_fitness_index]))


if __name__ == '__main__':
    print(ave_power(99994.63200537158, 0.9781081365411363))
    print(ave_power(99545.473, 0.99392718))
    print(ave_power(99999.39, 0.97834127))
    # pop 表示种群矩阵，一行表示一个二进制编码表示的DNA， 矩阵的行数为种群数目， DNA_SIZE为编码长度
    # pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
    # count = 0
    # for _ in range(N_GENERATIONS): # 种群迭代进化 N_GENERATIONS 代
    #     print(count)
    #     count+=1
    #     pop = np.array(crossover_and_mutation(pop)) # 种群通过交叉变异产生后代
    #     fitness = get_fitness(pop) # 对种群中每个个体进行评估
    #     pop = select(pop, fitness) # 选择产生新的种群
    #     print_info(pop)
    

    
