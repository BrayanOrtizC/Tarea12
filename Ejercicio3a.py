from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

def ODE_euler(
    *,
    a: float,
    b: float,
    f: Callable[[float, float], float],
    y_t0: float,
    N: int,
) -> tuple[list[float], list[float], float]:
    
    h = (b - a) / N #Se debe ajustar N para que h = 0.1
    t = a
    ts = [t]
    ys = [y_t0]

    for _ in range(N):
        y = ys[-1]
        y += h * f(t, y)
        ys.append(y)
        t += h
        ts.append(t)
    return ys, ts, h

f = lambda t, y: y/t - (y/t)**2
y_t0 = 1

a = 1
b = 2

ys,ts, h = ODE_euler(a=a, b=b, y_t0=y_t0, f=f, N=10)
print(ys)


# Solución real de la ecuación diferencial
# (No es parte del ejercicio pero la pongo para tener una idea)
# Parámetros del problema
h=0.1
n = int((b - a) / h) + 1 #numero de pasos

def solucion_real(t):
    return t / (1 + np.log(t))
t_values = np.linspace(a, b, n)
y_values = np.zeros(n)

# Calculamos la solución real en los puntos t_values
sol_real = solucion_real(t_values)

#Graficar
plt.plot(t_values, sol_real, 'r-', label="Solución real")

plt.plot(ts, ys, marker="o", linestyle=":", label=r"$\hat{y}$")
plt.xlabel("Time")
plt.ylabel("Y")
plt.title("Solución usando Método de Euler")
plt.legend()
plt.show()
