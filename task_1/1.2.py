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
    if "dimension" in ax.get_xlabel():
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 18])
    if ax.get_legend() is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                  ncol=min(3, len(labels)), frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(figure_dir / f"{name}.png", dpi=200, bbox_inches="tight")

rng = np.random.default_rng(5)

def clean_signal(x):
    return np.sin(2 * x * np.pi) ** 2

def sample_data(n):
    x_data = rng.uniform(0, 1, n)
    noise = rng.normal(0, 0.07, n)
    y_data = clean_signal(x_data) + noise
    return x_data, y_data

x, y = sample_data(30)

x_plot = np.linspace(0, 1, 500)

plt.plot(x_plot, clean_signal(x_plot), label=r"Clean signal: $\sin^2(2\pi x)$")
plt.scatter(x, y, color="black", s=22, zorder=3, label="30 noisy observations")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Clean signal and noisy training observations")
plt.legend()
save_figure("lr-q2-data")
plt.show()


def gen_poly_features(k, x_):
    columns = list()
    for power in range(0, k):
        columns.append(x_ ** power)
    return np.column_stack(columns)


def fit_data(phi, y):
    left = phi.T @ phi
    right = phi.T @ y
    return np.linalg.solve(left, right)


polynomials_to_plot = [2, 5, 10, 14, 18]

plt.figure()
plt.scatter(x, y, color="black", label="Data", zorder=3)

for k in polynomials_to_plot:
    features = gen_poly_features(k, x)
    w = fit_data(features, y)

    plot_features = gen_poly_features(k, x_plot)
    y_plot = plot_features @ w
    plt.plot(x_plot, y_plot, label=f"k={k}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial fits to the training observations")
plt.legend()
save_figure("lr-q2-fits")
plt.show()

training_errors = []
dimensions = range(1, 19)

for k in dimensions:
    features = gen_poly_features(k, x)
    w = fit_data(features, y)

    y_estimated = features @ w
    mse = np.mean((y_estimated - y) ** 2)
    training_errors.append(mse)

plt.figure()
plt.plot(dimensions, np.log(training_errors), marker="o")
plt.xlabel("Polynomial basis dimension $k$")
plt.ylabel("ln(training MSE)")
plt.title("Polynomial basis: training error (one run)")
plt.grid(alpha=0.3)
plt.tight_layout()
save_figure("lr-q2-train")
plt.show()


xt, yt = sample_data(1000)

test_errors = []
dimensions = range(1, 19)

for k in dimensions:
    features_t = gen_poly_features(k, xt)
    features = gen_poly_features(k, x)
    w = fit_data(features, y)

    y_estimated = features_t @ w
    mse = np.mean((y_estimated - yt) ** 2)
    test_errors.append(mse)

plt.figure()
plt.plot(dimensions, np.log(test_errors), marker="o")
plt.xlabel("Polynomial basis dimension $k$")
plt.ylabel("ln(test MSE)")
plt.title("Polynomial basis: test error (one run)")
plt.grid(alpha=0.3)
plt.tight_layout()
save_figure("lr-q2-test")
plt.show()

all_training_errors = []
all_test_errors = []

for i in range(100):
    training_errors = []
    dimensions = range(1, 19)
    x, y = sample_data(30)

    for k in dimensions:
        features = gen_poly_features(k, x)
        w = fit_data(features, y)

        y_estimated = features @ w
        mse = np.mean((y_estimated - y) ** 2)
        training_errors.append(mse)

    xt, yt = sample_data(1000)

    test_errors = []
    dimensions = range(1, 19)

    for k in dimensions:
        features_t = gen_poly_features(k, xt)
        features = gen_poly_features(k, x)
        w = fit_data(features, y)

        y_estimated = features_t @ w
        mse = np.mean((y_estimated - yt) ** 2)
        test_errors.append(mse)
    all_training_errors.append(training_errors)
    all_test_errors.append(test_errors)

mean_training_errors = np.mean(all_training_errors, axis=0)
mean_test_errors = np.mean(all_test_errors, axis=0)

plt.figure()
plt.plot(dimensions, np.log(mean_training_errors), marker="o")
plt.xlabel("Polynomial basis dimension $k$")
plt.ylabel("ln(mean training MSE)")
plt.title("Polynomial basis: training error (100 runs)")
plt.grid(alpha=0.3)
plt.tight_layout()
save_figure("lr-q2-mean-train")
plt.show()

plt.figure()
plt.plot(dimensions, np.log(mean_test_errors), marker="o")
plt.xlabel("Polynomial basis dimension $k$")
plt.ylabel("ln(mean test MSE)")
plt.title("Polynomial basis: test error (100 runs)")
plt.grid(alpha=0.3)
plt.tight_layout()
save_figure("lr-q2-mean-test")
plt.show()
