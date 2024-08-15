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
    
    h = (b - a) / N #Se debe ajustar N para que h = 0.5
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

def calcular_error(sol_real: np.ndarray, sol_aprox: np.ndarray) -> np.ndarray:
    return np.abs(sol_real - sol_aprox)

f = lambda t, y: np.cos(2*t) + np.sin(3*t)
y_t0 = 0

a = 0
b = 1

ys, ts, h = ODE_euler(a=a, b=b, y_t0=y_t0, f=f, N=2)
print("Solución aproximada:", ys)

# Solución real de la ecuación diferencial
h = 0.5
n = int((b - a) / h) + 1 # numero de pasos

def solucion_real(t):
    return (1/2) * np.sin(2*t) - (1/3) * np.cos(3*t) + (4/3)

t_values = np.linspace(a, b, n)
sol_real = solucion_real(t_values)

# Calcular error
error = calcular_error(sol_real, ys)
print("Error:", error)


# Gráfica del error
plt.subplot(1, 2, 2)
plt.plot(ts, error, marker="o", linestyle=":", label="Error absoluto")
plt.xlabel("Tiempo")
plt.ylabel("Error")
plt.title("Error absoluto: solución real vs aproximada")
plt.legend()

plt.show()
