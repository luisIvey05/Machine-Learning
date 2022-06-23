import numpy as np


def perceptron(X, y):
    (m, d) = X.shape
    X_tilde = np.insert(X, 0, np.ones(m), axis=1)  # Insert ones as the bias to the dataset
    w = np.zeros((d+1, m))
    w_temp = np.zeros((d+1, 1))

    for i, (x_i, y_i) in enumerate(zip(X_tilde, y)):
        if y_i * (x_i @ w_temp) <= 0:
            w_temp = (w_temp.T + y_i * x_i).T
        w[:, i] = w_temp.T

    b_last = w[0, i]
    w_last = w[1:, i]
    b_avg = w[0, :]
    w_avg = w[1:, :]
    b_avg = np.mean(b_avg, axis=0)
    w_avg = np.mean(w_avg, axis=0)

    return w_last, b_last, w_avg, b_avg


np.random.seed(0)
X = np.random.randn(9, 6)
y = np.random.randn(9, 1)

(w, b, w_avg, b_avg) = perceptron(X, y)

print(w)
print(b)
print(w_avg)
print(b_avg)