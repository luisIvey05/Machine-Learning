import numpy as np

def generate_poly_features(X, k):
    (m, d) = X.shape
    cntr = 2

    for x_i in X:
        X = np.vstack([X, abs(x_i)**(1/cntr)])
        if cntr < k:
            cntr += 1

np.random.seed(0)
X = np.random.randn(9, 6)

generate_poly_features(X, 10)
