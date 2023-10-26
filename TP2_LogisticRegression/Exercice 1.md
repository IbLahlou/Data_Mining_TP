```python
import numpy as np

import matplotlib.pyplot as plt

  

def plot_decision_boundary(X, y, bias, w1, w2):

    # Tracer le graphe

    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', marker='o')

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', marker='x')

  

    # Ligne de Spération / Decision Boundary

    x_values = np.array([np.min(X[:, 0]), np.max(X[:, 0])])

    y_values = (-1/w2) * (bias + w1 * x_values)

    plt.plot(x_values, y_values, label='Decision Boundary', color='red')

  

    plt.xlabel('w1')

    plt.ylabel('w2')

    plt.legend()

  
  

from utils import *  

  

def sigmoid(Z):

    return 1 / (1 + np.exp(-Z))

  

def log_loss(y, yhat):

    epsilon = 1e-15

    yhat = np.maximum(epsilon, yhat)

    yhat = np.minimum(1 - epsilon, yhat)

    return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()

  

def fit(X, y):

    lr = 0.01

    epochs = 100000

    n = len(y)

    w1 = 0.5

    w2 = 0.5

    bias = 0.5

  

    for k in range(epochs):

        ws = bias + w1 * X[:, 0] + w2 * X[:, 1]

        yhat = sigmoid(ws)

  

        cost = log_loss(y, yhat)

  

        dbias = -(2 / n) * np.sum(y - yhat)

        dw1 = -(2 / n) * np.sum(X[:, 0] * (y - yhat))

        dw2 = -(2 / n) * np.sum(X[:, 1] * (y - yhat))

  

        bias = bias - dbias * lr

        w1 = w1 - dw1 * lr

        w2 = w2 - dw2 * lr

  

        if k in [0, 100, 500, 1700, 8000]:

            plt.figure()

            plot_decision_boundary(X, y, bias, w1, w2)

            plt.title(f'Decision Boundary at Iteration {k}')

  

            plt.figure()

            plt.plot(range(k + 1), [log_loss(y, sigmoid(bias + w1 * X[:, 0] + w2 * X[:, 1])) for k in range(k + 1)])

            plt.xlabel('Iteration')

            plt.ylabel('Cost')

            plt.title(f'Cost at Iteration {k}')

            plt.show()

  

    return bias, w1, w2

  

bias, w1, w2 = fit(X, C)
```