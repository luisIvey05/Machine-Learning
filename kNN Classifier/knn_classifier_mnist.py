import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(root = './', train=True, transform=transform, download=False)
mnist_test_dataset = torchvision.datasets.MNIST(root = './', train=False, transform=transform, download=False)
mnist_train_dataset = Subset(mnist_dataset, torch.arange(500))
mnist_test_dataset = Subset(mnist_test_dataset, torch.arange(100))

batch_size = 10
torch.manual_seed(1)
k = 1
confusion_matrix = np.zeros((10, 10))
for x_batch, y_batch in mnist_test_dataset:
    x_batch = x_batch.numpy().flatten()
    temp_dist = np.zeros((500, 2))
    for idx, (x_train, y_train) in enumerate(mnist_train_dataset):
        x_train = x_train.numpy().flatten()
        temp_dist[idx] = (np.linalg.norm(x_batch - x_train), y_train)
    dist = sorted(temp_dist, key=lambda x: x[0])[:k]
    dist = np.stack(dist, axis=0)
    confusion_matrix[int(dist[:, 1]), y_batch] += 1

print(confusion_matrix)
