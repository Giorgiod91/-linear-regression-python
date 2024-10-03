import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300, 500])

w= 200
b= 100
x_i =1.2

cost_example = w * x_i + b 
print(cost_example)

def compute_model_output(x,w,b):
    # x (ndarray (m,)) : Data, m examples
    # w, b (scalar (m,)) : Model parameters
    # Return f_wb (ndarray (m,)) : Model prediction

    m= x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    return f_wb



# Number of training samples called m
m = x_train.shape[0]
print(m);

tmp_f_wb = compute_model_output(x_train, w, b)
# Plotting model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Model Prediction')


# Plotting the data

plt.scatter(x_train, y_train, marker='x', color='r', label='Actual Values')
plt.title('Training Data')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.legend()
plt.show()


