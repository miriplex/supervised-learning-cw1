import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Use the same readable figure style across Section 1.1.
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (6.4, 3.2),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
})

figure_dir = Path(__file__).resolve().parents[1] / "output" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)


def save_figure(name):
    """Export a sharp vector figure and a PNG preview without changing its data."""
    ax = plt.gca()
    if ax.get_legend() is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                  ncol=min(3, len(labels)), frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(figure_dir / f"{name}.png", dpi=200, bbox_inches="tight")

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
plt.figure()
plt.scatter(x, y, color="black", label="Data", zorder=3)

plt.plot(x_plot, y1_plot, label="k=1: constant")
plt.plot(x_plot, y2_plot, label="k=2: straight line")
plt.plot(x_plot, y3_plot, label="k=3: quadratic")
plt.plot(x_plot, y4_plot, label="k=4: cubic")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial fits to four data points")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_figure("lr-q1-fits")
plt.show()

print(w1)
print(w2)
print(w3)
print(w4)

fits = [
    (phi_k1, w1),
    (phi_k2, w2),
    (phi_k3, w3),
    (phi_k4, w4),
]

for k, (phi, w) in enumerate(fits):
    predictions = phi @ w
    errors = predictions - y
    mse = np.mean(np.square(errors))
    print(f"k={k + 1}: MSE={mse}")
