import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([250, 265, 272, 260, 248, 240, 230, 220, 210, 200])




# Number of training samples called m
m = x_train.shape[0]
print(m);

# Plotting the data

plt.scatter(x_train, y_train, marker='x', color='r')
plt.title('Training Data')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.show()