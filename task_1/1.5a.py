from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, StrMethodFormatter


from kernel_ridge import GAMMA_VALUES, SIGMA_VALUES, cross_validation_errors, fit_data, gaussian_kernel


data_path = Path(__file__).resolve().parents[1] / "data" / "Boston-filtered.csv"
data = np.loadtxt(data_path, delimiter=",", skiprows=1)

X = data[:, :-1]
y = data[:, -1]

rng = np.random.default_rng(5)
n_train = int(2 * len(X) / 3)
indices = rng.permutation(len(X))
train_indices = indices[:n_train]
X_train = X[train_indices]
y_train = y[train_indices]

cv_errors = cross_validation_errors(X_train, y_train)
best_i, best_j = np.unravel_index(np.argmin(cv_errors), cv_errors.shape)
best_gamma = GAMMA_VALUES[best_i]
best_sigma = SIGMA_VALUES[best_j]

K_train = gaussian_kernel(X_train, X_train, best_sigma)
alpha = fit_data(K_train, y_train, best_gamma)

print(f"Best gamma: {best_gamma:.6g}")
print(f"Best sigma: {best_sigma:.6g}")
print(f"Mean validation MSE: {cv_errors[best_i, best_j]:.4f}")

# Q5(c): Predict prices using the model fitted on all training rows.
test_indices = indices[n_train:]
X_test = X[test_indices]
y_test = y[test_indices]
K_test = gaussian_kernel(X_test, X_train, best_sigma)

y_train_estimated = K_train @ alpha
y_test_estimated = K_test @ alpha
mse_train = np.mean((y_train_estimated - y_train) ** 2)
mse_test = np.mean((y_test_estimated - y_test) ** 2)

print(f"Training MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# Q5(b): Each cell shows the mean MSE across the five validation folds.
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11})
fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
heatmap = ax.imshow(cv_errors, origin="lower", aspect="auto", cmap="viridis", norm=LogNorm())

sigma_ticks = np.arange(0, len(SIGMA_VALUES), 2)
gamma_ticks = np.arange(0, len(GAMMA_VALUES), 2)
ax.set_xticks(sigma_ticks, [f"{value:g}" for value in np.log2(SIGMA_VALUES[sigma_ticks])])
ax.set_yticks(gamma_ticks, [f"{value:g}" for value in np.log2(GAMMA_VALUES[gamma_ticks])])
ax.set_xlabel(r"$\log_2 \sigma$")
ax.set_ylabel(r"$\log_2 \gamma$")
ax.set_title("Five-fold cross-validation")
ax.scatter(best_j, best_i, s=70, facecolors="none", edgecolors="white", linewidths=1.5)

colorbar = fig.colorbar(
    heatmap, ax=ax, label="Mean validation MSE (log scale)",
    ticks=LogLocator(base=10, subs=(1, 2, 5)), format=StrMethodFormatter("{x:g}"),
)
colorbar.minorticks_off()

figure_dir = Path(__file__).resolve().parents[1] / "output" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_dir / "krr-q5-cv.pdf", bbox_inches="tight")
fig.savefig(figure_dir / "krr-q5-cv.png", dpi=200, bbox_inches="tight")
plt.show()
