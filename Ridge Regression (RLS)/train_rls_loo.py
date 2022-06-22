import numpy as np

def train_rls_loo(X, y, lambdaa, epsilon):
    (m, d) = X.shape
    X_tilde = np.insert(X, 0, np.ones(m), axis=1)  # Insert ones as the bias to the dataset
    Q = np.zeros((d + 1, d + 1))  # Construct Q = [epsilon, zeros row; zero column, lambda * eye(d)]
    Q[0, 0] = epsilon
    Q[1:, 1:] = lambdaa * np.eye(d)
    C = X_tilde.T @ X_tilde + Q
    C_inv = np.linalg.pinv(C)  # Invert C (even if C is not invertible)
    w = C_inv @ (X_tilde.T @ y)
    b = w[0]


    train_err = np.mean((X_tilde @ w - y)**2, axis=0)
    loo_err = np.zeros((m, 1))

    for i, (x_i, y_i) in enumerate(zip(X_tilde, y)):
        loo_err[i] = (x_i @ w - y_i) / (1 - x_i @ C_inv @ x_i)
    loo_err = np.mean(loo_err, axis=1)

    w = w[1:]

    return w, b, train_err, loo_err


np.random.seed(0)
X = np.random.randn(9, 6)
y = np.random.randn(9, 1)
lambdaa = 0.0001
epsilon = 0.3

(w, b, train_err, test_err) = train_rls_loo(X, y, lambdaa, epsilon)

print(w)
print(b)
print(train_err)
print(test_err)