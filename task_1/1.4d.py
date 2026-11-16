from pathlib import Path

import numpy as np

data_path = Path(__file__).resolve().parents[1] / "data" / "Boston-filtered.csv"

with data_path.open(encoding="utf-8") as csv_file:
    column_names = [name.strip() for name in csv_file.readline().split(",")]

data = np.loadtxt(data_path, delimiter=",", skiprows=1)

X = data[:, :-1]
y = data[:, -1]
feature_names = column_names[:-1]
target_name = column_names[-1]

rng = np.random.default_rng(5)
n_train = int(2 * len(X) / 3)

def fit_data(phi, y):
    left = phi.T @ phi
    right = phi.T @ y
    return np.linalg.solve(left, right)


mse_train_set = list()
mse_test_set = list()
for i in range(20):
    indices = rng.permutation(len(X))

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    # Each run contributes 12 errors, in the order given by feature_names.

    # Use one attribute and a separate column of ones for the bias.
    features_train = np.column_stack((
        X_train,
        np.ones(len(X_train))
    ))
    features_test = np.column_stack((
        X_test,
        np.ones(len(X_test))
    ))

    w = fit_data(features_train, y_train)
    y_test_estimated = features_test @ w
    y_train_estimated = features_train @ w

    mse_train = np.mean((y_train_estimated - y_train) ** 2)
    mse_test = np.mean((y_test_estimated - y_test) ** 2)
    mse_train_set.append(mse_train)
    mse_test_set.append(mse_test)

mean_train_mse = np.mean(mse_train_set)
mean_test_mse = np.mean(mse_test_set)

print(mean_train_mse)
print(mean_test_mse)