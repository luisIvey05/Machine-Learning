import numpy as np


def train_rls(X, y, lambdaa, epsilon):
    (m, d) = X.shape
    X_tilde = np.insert(X, 0, np.ones(m), axis=1)  # Insert ones as the bias to the dataset
    Q = np.zeros((d + 1, d + 1))  # Construct Q = [epsilon, zeros row; zero column, lambda * eye(d)]
    Q[0, 0] = epsilon
    Q[1:, 1:] = lambdaa * np.eye(d)
    C = X_tilde.T @ X_tilde + Q
    C_inv = np.linalg.pinv(C)  # Invert C (even if C is not invertible)
    w = C_inv @ (X_tilde.T @ y)
    b = w[0]
    w = w[1:]

    return w, b

np.random.seed(0)
X = np.random.randn(9, 6)
y = np.random.randn(9, 1)
lambdaa = 0.0001
epsilon = 0.3

(w, b) = train_rls(X, y, lambdaa, epsilon)

print(w)
print(b)