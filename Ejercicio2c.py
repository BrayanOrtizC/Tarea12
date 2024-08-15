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

f = lambda t, y: 1 + y/t
y_t0 = 2

a = 1
b = 2

ys,ts, h = ODE_euler(a=a, b=b, y_t0=y_t0, f=f, N=2)
print(ys)


# Solución real de la ecuación diferencial
# (No es parte del ejercicio pero la pongo para tener una idea)
# Parámetros del problema
h=0.5
n = int((b - a) / h) + 1 #numero de pasos

def solucion_real(t):
    return t * np.log(t) + 2*t
t_values = np.linspace(a, b, n)
y_values = np.zeros(n)

# Calculamos la solución real en los puntos t_values
sol_real = solucion_real(t_values)



# Calcular error

def calcular_error(sol_real: np.ndarray, sol_aprox: np.ndarray) -> np.ndarray:
    return np.abs(sol_real - sol_aprox)
error = calcular_error(sol_real, ys)
print("Error:", error)


# Gráfica del error
plt.plot(ts, error, marker="o", linestyle=":", label="Error absoluto")
plt.xlabel("Tiempo")
plt.ylabel("Error")
plt.title("Error absoluto: solución real vs aproximada")
plt.legend()

plt.show()
