from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor


b0 = 10000
k = 80000
m1 = 4866  # 浮子
m2 = 2433  # 振子
g = 9.8
r = 1

def get_A(v):
    b = b0 * np.sqrt(v)
    A = np.array([[0, 1, 0, 0],
              [-(k + np.pi * 1025 * 9.8 * r ** 2) /m1, -b/m1, k/m1, b/m1],
              [0, 0, 0, 1],
              [k/m2, b/m2, -k/m2, -b/m2]]
              )
    return A
# f1 = f2 = f3 = 5*np.sin(1.25*t)

B = np.array([[0, 0],
              [1/m1, 0],
              [0, 0],
              [0, 1/m2]
             ])

def func(t,y):
    #  dX/dt = A*X+b
    #  x1,x2,x3,x4,x5,x6 = y\
    u = np.array([6250*np.cos(1.4005 * t)-4866*9.8+8415.28, -2433*9.8])
    return [np.dot(A[i,:], y)+np.dot(B[i,:],u)  for i in range(4)]

N = 100
dt = 0.1
# t_span = (0,N)
# t_eval = np.linspace(0,N,1000)
y0 = [-1.798,0,-2,0]
A0 = get_A(0)

A = A0
x1 = [-1.798]
v1 = [0]
x2 = [-2]
v2 = [0]

for step in range(floor(N / dt)):
    start_t = step  * dt
    end_t = start_t + 10 * dt
    t_span = (start_t, end_t)
    t_eval = np.linspace(start_t, end_t, 10)
    
    y0 = [x1[-1], v1[-1], x2[-1], v2[-1]]
    sol = solve_ivp(func, t_span, y0, t_eval=t_eval)
    A = get_A(np.abs(sol.y.T[1, 1]-sol.y.T[1, 3]))
    
    x1.append(sol.y.T[1, 0])
    v1.append(sol.y.T[1, 1])
    x2.append(sol.y.T[1, 2])
    v2.append(sol.y.T[1, 3])

    
t = np.linspace(0, N + dt, floor(N / dt) + 1)
plt.subplot(411)
plt.plot(t, x1)
plt.subplot(412)
plt.plot(t, v1)
plt.subplot(413)
plt.plot(t, x2)
plt.subplot(414)
plt.plot(t, v2)
plt.show()