import numpy as np
import matplotlib.pyplot as plt

# Data points
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([3, 2, 0, 5], dtype=float)

ones_like = np.ones_like(x)
x2 = x ** 2
x3 = x ** 3
phi_k1 = np.ones((len(x), 1))
phi_k2 = np.column_stack((ones_like, x))
phi_k3 = np.column_stack((ones_like, x, x2))
phi_k4 = np.column_stack((ones_like, x, x2, x3))


def fit_data(phi, y):
    left = phi.T @ phi
    right = phi.T @ y
    return np.linalg.solve(left, right)

w1 = fit_data(phi_k1, y)
w2 = fit_data(phi_k2, y)
w3 = fit_data(phi_k3, y)
w4 = fit_data(phi_k4, y)

x_plot = np.linspace(0.5, 4.25, 200)

y1_plot = np.full_like(x_plot, w1[0])
y2_plot = w2[0] + w2[1] * x_plot
y3_plot = w3[0] + w3[1] * x_plot + w3[2] * x_plot ** 2
y4_plot = (
    w4[0]
    + w4[1] * x_plot
    + w4[2] * x_plot ** 2
    + w4[3] * x_plot ** 3
)

# Draw all four curves on the same axes
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="black", label="Data", zorder=3)

plt.plot(x_plot, y1_plot, label="k=1: constant")
plt.plot(x_plot, y2_plot, label="k=2: straight line")
plt.plot(x_plot, y3_plot, label="k=3: quadratic")
plt.plot(x_plot, y4_plot, label="k=4: cubic")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(w1)
print(w2)
print(w3)
print(w4)