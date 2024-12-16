import numpy as np
import data_extended as data
import lib
import scipy.integrate as spi
from matplotlib import pyplot as plt

vectors = [lib.ConvertArrFromBinary(v) for v in data.Base]
vectors = [lib.CenterVector(v) for v in vectors]
vectors = [lib.NormalizeVector(v) for v in vectors]
vectors = np.array(vectors)
print("Вектора:\n", vectors)
print("Суммы значений векторов:", [np.sum(v) for v in vectors])
print("Суммы абсолютных значений векторов:", [np.sum(np.abs(v)) for v in vectors])

v_plus = lib.FindPlusVectors(vectors)
print("Cопряжённые вектора:\n", v_plus)

q = lib.ConvertArrFromBinary(data.Test)
print("Тестируемый вектор: ", q)
q = lib.CenterVector(q)
q = lib.NormalizeVector(q)

p = np.array([np.dot(q, v_p) for v_p in v_plus])
print("Параметры порядка: ", p)

multiplyers = []
for i in range(len(p)):
    multiplyers.append([data.lambdas[i]*p[i],
               - data.B*p[i],
               - data.C*p[i]
               ])
    

def diff_equation(t, ps):
    res = np.zeros(len(ps))
    sum_p = (ps[0]**2 + ps[1]**2 + ps[2]**2)
    for i in range(len(ps)):
        res[i] = ps[i] - data.B*ps[i]*(sum_p - ps[i]**2) - data.C*ps[i]*sum_p
    return res

solution = spi.solve_ivp(diff_equation, [0, 5], p, t_eval=np.linspace(0,5,100))

plt.plot(solution.t, solution.y[0], label="L", color="red")
plt.plot(solution.t, solution.y[1], label="C", color="blue")
plt.plot(solution.t, solution.y[2], label="A", color="green")
plt.grid()
plt.legend()
plt.show()