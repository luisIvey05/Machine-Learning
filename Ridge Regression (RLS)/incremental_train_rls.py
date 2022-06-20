import numpy as np


def incremental_train_rls(X, y, lambdaa, epsilon):
    (m, d) = X.shape
    X_tilde = np.insert(X, 0, np.ones(m), axis=1)  # Insert ones as the bias to the dataset
    Q = np.zeros((d + 1, d + 1))  # Construct Q = [epsilon, zeros row; zero column, lambda * eye(d)]
    Q[0, 0] = 1/epsilon
    Q[1:, 1:] = (1/lambdaa) * np.eye(d)
    A_inv = Q
    X_primey = np.zeros(d+1) # X'*y with no current samples
    w = np.zeros(d+1)

    for (x_next, y_next) in zip(X_tilde, y):
        p_next = A_inv @ x_next
        A_inv = A_inv - np.outer(p_next, p_next) / (1 + x_next @ p_next)
        X_primey = X_primey + x_next * y_next
        w = w + p_next * y_next - np.outer(p_next, p_next) @ X_primey / (1 + x_next @ p_next)

    b = w[0]
    w = w[1:]

    return w, b


np.random.seed(0)
X = np.random.randn(9, 6)
y = np.random.randn(9, 1)
lambdaa = 0.0001
epsilon = 0.3

(w, b) = incremental_train_rls(X, y, lambdaa, epsilon)

print(w)
print(b)
