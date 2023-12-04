import numpy as np
import matplotlib.pyplot as plt

def viz(X,y):
    # Scatter plot
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='o', label='Class 1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b', marker='x', label='Class 0')

    # Set labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of XOR Dataset')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()