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

f = lambda t, y: -5*y + 5*t**2 + 2*t
y_t0 = 1/3

a = 0
b = 1

ys,ts, h = ODE_euler(a=a, b=b, y_t0=y_t0, f=f, N=10)
print(ys)


# Solución real de la ecuación diferencial
# (No es parte del ejercicio pero la pongo para tener una idea)
# Parámetros del problema
h=0.1
n = int((b - a) / h) + 1 #numero de pasos

def solucion_real(t):
    return t**2 + (1/3) * np.exp(-5*t)
t_values = np.linspace(a, b, n)
y_values = np.zeros(n)

# Calculamos la solución real en los puntos t_values
sol_real = solucion_real(t_values)

#Calcular interpolación

def linear_interpolation(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def interpolate_and_compare(ts, ys, exact_solution, points):
    results = []
    for t in points:
        # Encontrar los puntos entre los que se va a interpolar
        for i in range(len(ts) - 1):
            if ts[i] <= t <= ts[i+1]:
                approx = linear_interpolation(ts[i], ys[i], ts[i+1], ys[i+1], t)
                exact = exact_solution(t)
                error = abs(approx - exact)
                results.append((t, approx, exact, error))
                break
    return results
# Puntos a interpolar
points_a = [0.25, 0.93]

# Interpolación y comparación
results_a = interpolate_and_compare(ts, ys, solucion_real, points_a)

# Mostrar resultados
for t, approx, exact, error in results_a:
    print(f"t = {t:.2f}, Error: {error:.5f}")


#Graficar
for t, approx, exact, _ in results_a:
    plt.plot([t, t], [exact, approx], 'g--')  # Línea entre la solución real y la aproximación

# Mostrar los puntos exactos y aproximados
exact_ts = [t for t, _, exact, _ in results_a]
exact_ys = [exact for _, _, exact, _ in results_a]
approx_ys = [approx for _, approx, _, _ in results_a]

plt.plot(exact_ts, exact_ys, 'ro', label="Solución exacta en puntos interpolados")
plt.plot(exact_ts, approx_ys, 'bo', label="Aproximación en puntos interpolados")


plt.plot(t_values, sol_real, 'r-', label="Solución real")

plt.plot(ts, ys, marker="o", linestyle=":", label=r"$\hat{y}$")
plt.xlabel("Time")
plt.ylabel("Y")
plt.title("Solución usando Método de Euler")
plt.legend()
plt.show()
