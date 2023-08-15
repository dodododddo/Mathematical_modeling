from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


b = 10000
k = 80000
m1 = 4866  # 浮子
m2 = 2433  # 振子
g = 9.8
r = 1

# f1 = f2 = f3 = 5*np.sin(1.25*t)
A = np.array([[0, 1, 0, 0],
              [-(k+np.pi*1025*9.8) * r ** 2 /m1, -b/m1, k/m1, b/m1],
              [0, 0, 0, 1],
              [k/m2, b/m2, -k/m2, -b/m2]]
              )
B = np.array([[0, 0],
              [1/m1, 0],
              [0, 0],
              [0, 1/m2]
             ])
# solve dX/dt = A*X+b
def func(t,y):
    #dX/dt = A*X+b
#     x1,x2,x3,x4,x5,x6 = y
    u = np.array([6250*np.cos(1.4005*t)-4866*9.8+8415.28, -2433*9.8])
    return [np.dot(A[i,:], y)+np.dot(B[i,:],u)  for i in range(4)]



N = 100
t_span = (0,N)
t_eval = np.linspace(0,N,1000)
y0 = [-1.798,0,-2,0]


sol = solve_ivp(func, t_span, y0, t_eval=t_eval)




print(len(sol.t))
plt.subplot(411)
plt.plot(sol.t, sol.y.T[:,0])
plt.subplot(412)
plt.plot(sol.t, sol.y.T[:,1])
plt.subplot(413)
plt.plot(sol.t, sol.y.T[:,2])
plt.subplot(414)
plt.plot(sol.t, sol.y.T[:,3])
true = np.sqrt(np.abs(sol.y.T[:,1]-sol.y.T[:,3]))
test1 = (5 / (2 ** 0.5)) * np.abs((sol.y.T[:,1] - sol.y.T[:,3]) - (2 ** 0.5 / 20))




# plt.plot(sol.t, sol.y.T[:,2]))

plt.show()
