import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def plot_data(data):
    red = (0, 0, 1)
    green = (0, 1, 0)
    blue = (1, 0, 0)
    colors = [red, green, blue]

    for idx, cls in enumerate(data):
        plt.scatter(
            x=cls[:, 0],
            y=cls[:, 1],
            color=colors[idx],
        )
        plt.xlim(-3.5, 6)
        plt.ylim(-3, 6.5)
        plt.legend(('blue', 'green', 'red'), loc='upper left')
    plt.show()

def heatmap(grid):
    plt.pcolormesh(grid, cmap="RdYlGn")
    plt.title("2-D Probability Map")
    plt.colorbar()
    plt.show()


def colormap(grid):
    bounds = np.array([0, 1.1, 2.1, 3.1])
    cmap = colors.ListedColormap(['blue', 'green', 'red'])
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.pcolormesh(grid, cmap=cmap, norm=norm)
    print()
    print(grid)
    plt.title("2-D Decision Map")
    plt.colorbar()
    plt.show()


def probmap(data, x, k, cls):
    temp_dist = np.zeros((len(data), 2))
    for idx, curr_pt in enumerate(x):
        for jdx, other_pt in enumerate(data):
            temp_dist[jdx] = (np.linalg.norm(curr_pt[:2] - other_pt[:2]), other_pt[2])
        dist = sorted(temp_dist, key=lambda x:x[0])[:k]
        dist = np.stack(dist, axis=0)
        x[idx, 2] = np.count_nonzero(dist[:, 1] == cls) / k
    grid = x[:, 2].reshape((10, 10))
    heatmap(grid)

def decmap(data, x, k):
    temp_dist = np.zeros((len(data), 2))
    for idx, curr_pt in enumerate(x):
        for jdx, other_pt in enumerate(data):
            temp_dist[jdx] = (np.linalg.norm(curr_pt[:2] - other_pt[:2]), other_pt[2])
        dist = sorted(temp_dist, key=lambda x: x[0])[:k]
        dist = np.stack(dist, axis=0)
        if k == 1:
            x[idx, 2] = int(dist[:, 1])
        else:
            x[idx, 2] = np.bincount(np.int64(dist[:, 1])).argmax()
    grid = x[:, 2].reshape((10, 10))
    colormap(grid)


def loocv_ccr(X, test, k):
    temp_dist = np.zeros((len(X), 2))
    for idx, curr_pt in enumerate(X):
        temp_dist[idx] = (np.linalg.norm(curr_pt[:2] - test[:2]), curr_pt[2])
    dist = sorted(temp_dist, key=lambda x: x[0])[:k]
    dist = np.stack(dist, axis=0)
    if k == 1:
        p_hat = int(dist[:, 1])
    else:
        p_hat = np.bincount(np.int64(dist[:, 1])).argmax()

    if p_hat == test[2]:
        return 1
    else:
        return 0


def loocv(data, ks):
    results = np.zeros((len(ks), 1))
    for kdx, k in enumerate(ks):
        for idx, loo in enumerate(data):
            temp_data = data
            temp_data = np.delete(data, idx, 0)
            results[kdx] = results[kdx] + loocv_ccr(temp_data, loo, k)
        results[kdx] = np.mean(results, axis=0)
        print(results)
        print()

    plt.title("LOOCV-CCR")
    plt.plot(ks, results, color="red")

    plt.show()



np.random.seed(5)
class_A = np.random.randn(20, 2)
class_A = np.append(class_A, np.ones((20,1)), axis=1)
class_B = np.random.randn(20, 2)
class_B = np.append(class_B, 2*np.ones((20, 1)), axis=1)
class_C = np.random.randn(20, 2)
class_C = np.append(class_C, 3*np.ones((20, 1)), axis=1)

data = [class_A, class_B, class_C]
plot_data(data)

data = np.append(class_A, class_B, axis=0)
data = np.append(data, class_C, axis=0)
# x_axis = np.arange(-3.5, 6)
# y_axis = np.arange(-3, 6.5)
# x = np.array(np.meshgrid(x_axis, y_axis)).T.reshape(-1, 2)
# x = np.append(x, np.zeros((len(x), 1)), axis=1)
# k = 10
# probmap(data, x, k, 2)
#
# k = 1
# decmap(data, x, k)
#
# k = 5
# decmap(data, x, k)

ks = [1, 3, 5, 7, 9, 11]
loocv(data, ks)

