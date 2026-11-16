import csv
from pathlib import Path

import numpy as np

from kernel_ridge import GAMMA_VALUES, SIGMA_VALUES, cross_validation_errors, fit_data, gaussian_kernel


project_dir = Path(__file__).resolve().parents[1]
data_path = project_dir / "data" / "Boston-filtered.csv"
with data_path.open(encoding="utf-8") as csv_file:
    column_names = [name.strip() for name in csv_file.readline().split(",")]

data = np.loadtxt(data_path, delimiter=",", skiprows=1)
X = data[:, :-1]
y = data[:, -1]
feature_names = column_names[:-1]

method_names = ["Constant baseline"]
method_names += [f"{name} + bias" for name in feature_names]
method_names += ["All 12 attributes + bias", "Kernel ridge regression"]

# Choose which columns of the augmented feature matrix each linear model uses.
bias_column = X.shape[1]
linear_columns = [[bias_column]]
linear_columns += [[j, bias_column] for j in range(bias_column)]
linear_columns.append(list(range(bias_column + 1)))

rng = np.random.default_rng(5)
n_train = int(2 * len(X) / 3)
n_runs = 20
train_errors = []
test_errors = []

for _ in range(n_runs):
    indices = rng.permutation(len(X))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    features_train = np.column_stack((X_train, np.ones(len(X_train))))
    features_test = np.column_stack((X_test, np.ones(len(X_test))))
    train_errors_this_run = []
    test_errors_this_run = []

    for columns in linear_columns:
        phi_train = features_train[:, columns]
        phi_test = features_test[:, columns]
        w = np.linalg.solve(phi_train.T @ phi_train, phi_train.T @ y_train)

        train_errors_this_run.append(np.mean((phi_train @ w - y_train) ** 2))
        test_errors_this_run.append(np.mean((phi_test @ w - y_test) ** 2))

    # Tune on this split's training rows, then refit on all of them.
    cv_errors = cross_validation_errors(X_train, y_train)
    best_i, best_j = np.unravel_index(np.argmin(cv_errors), cv_errors.shape)
    best_gamma = GAMMA_VALUES[best_i]
    best_sigma = SIGMA_VALUES[best_j]

    K_train = gaussian_kernel(X_train, X_train, best_sigma)
    K_test = gaussian_kernel(X_test, X_train, best_sigma)
    alpha = fit_data(K_train, y_train, best_gamma)
    train_errors_this_run.append(np.mean((K_train @ alpha - y_train) ** 2))
    test_errors_this_run.append(np.mean((K_test @ alpha - y_test) ** 2))

    train_errors.append(train_errors_this_run)
    test_errors.append(test_errors_this_run)

# Each column represents one model. Summarise its errors across the 20 splits.
train_mean = np.mean(train_errors, axis=0)
test_mean = np.mean(test_errors, axis=0)
train_std = np.std(train_errors, axis=0, ddof=1)
test_std = np.std(test_errors, axis=0, ddof=1)

results_dir = project_dir / "output" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
results_path = results_dir / "q5-model-comparison.csv"

print(f"Mean MSE ± sample standard deviation over {n_runs} splits")
print(f"{'Method':<27} {'Train MSE':>21} {'Test MSE':>21}")
with results_path.open("w", encoding="utf-8", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["method", "train_mean", "train_std", "test_mean", "test_std"])
    for name, tr_mean, tr_std, te_mean, te_std in zip(
        method_names, train_mean, train_std, test_mean, test_std
    ):
        writer.writerow([name, tr_mean, tr_std, te_mean, te_std])
        print(f"{name:<27} {tr_mean:>9.4f} ± {tr_std:<9.4f} {te_mean:>9.4f} ± {te_std:<9.4f}")
