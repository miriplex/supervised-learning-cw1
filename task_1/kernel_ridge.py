import numpy as np


GAMMA_VALUES = 2.0 ** np.arange(-40, -25)
SIGMA_VALUES = 2.0 ** np.arange(7, 13.5, 0.5)


def gaussian_kernel(X_left, X_right, sigma):
    differences = X_left[:, None, :] - X_right[None, :, :]
    squared_distances = np.sum(differences ** 2, axis=2)
    return np.exp(-squared_distances / (2 * sigma ** 2))


def fit_data(K, y, gamma):
    n = len(y)
    return np.linalg.solve(K + gamma * n * np.eye(n), y)


def cross_validation_errors(X_train, y_train):
    folds = np.array_split(np.arange(len(X_train)), 5)
    cv_errors = np.zeros((len(GAMMA_VALUES), len(SIGMA_VALUES)))

    for j, sigma in enumerate(SIGMA_VALUES):
        K_train = gaussian_kernel(X_train, X_train, sigma)

        for i, gamma in enumerate(GAMMA_VALUES):
            fold_errors = []

            for fold in range(len(folds)):
                val_indices = folds[fold]
                fit_indices = np.concatenate(folds[:fold] + folds[fold + 1:])

                K_fit = K_train[np.ix_(fit_indices, fit_indices)]
                K_val = K_train[np.ix_(val_indices, fit_indices)]

                alpha = fit_data(K_fit, y_train[fit_indices], gamma)
                y_val_estimated = K_val @ alpha
                mse_val = np.mean((y_val_estimated - y_train[val_indices]) ** 2)
                fold_errors.append(mse_val)

            cv_errors[i, j] = np.mean(fold_errors)

    return cv_errors
