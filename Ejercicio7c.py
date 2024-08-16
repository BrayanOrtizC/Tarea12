from math import factorial
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def ODE_euler_nth(
    *,
    a: float,
    b: float,
    f: Callable[[float, float], float],
    f_derivatives: list[Callable[[float, float], float]],
    y_t0: float,
    N: int,
) -> tuple[list[float], list[float], float]:

    h = (b - a) / N
    print(f"h = {h}")
    t = a
    ts = [t]
    ys = [y_t0]

    for _ in range(N):
        y = ys[-1]
        T = f(t, y)
        ders = [
            h ** (m + 1) / factorial(m + 2) * mth_derivative(t, y)
            for m, mth_derivative in enumerate(f_derivatives)
        ]
        T += sum(ders)
        y += h * T
        ys.append(y)

        t += h
        ts.append(t)
    return ys, ts, h

# Definir la función y sus derivadas
f = lambda t, y: 1 + y / t
f_p = lambda t, y: 1 / t
f_2p = lambda t, y: -1 / t**2
f_3p = lambda t, y: 2 / t**3


y_t0 = 2

a = 1
b = 2

# Usar el método de Taylor de 4to orden
ys_nth, ts_nth, h = ODE_euler_nth(
    a=a, b=b, y_t0=y_t0, f=f, N=4, f_derivatives=[f_p, f_2p, f_3p]
)
print(ys_nth)

# Solución real de la ecuación diferencial
h = 0.25
n = int((b - a) / h) + 1  # número de pasos

def solucion_real(t):
    return t * np.log(t) + 2*t
t_values = np.linspace(a, b, n)

# Calculamos la solución real en los puntos t_values
sol_real = solucion_real(t_values)

# Graficar
plt.plot(t_values, sol_real, 'r-', label="Solución real")
plt.plot(ts_nth, ys_nth, marker="o", linestyle=":", label="Método de Taylor de 4to orden")
plt.xlabel("Tiempo")
plt.ylabel("Y")
plt.title("Método de Taylor de 4to orden")
plt.legend()
plt.show()
