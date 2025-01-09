import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("deeplearning.mpstyle")
np.set_printoptions(precision=2)



X_train = np.array([
    [2104, 5, 1, 45],    # House 1
    [1416, 3, 2, 40],    # House 2
    [852, 2, 1, 35],     # House 3
    [1800, 4, 2, 30],    # House 4
    [1700, 3, 2, 15],    # House 5
    [2350, 5, 3, 5],     # House 6
    [2200, 4, 3, 20],    # House 7
    [1850, 3, 2, 10],    # House 8
    [2100, 5, 3, 25],    # House 9
    [1500, 3, 2, 5]      # House 10
])
y_train = np.array([460, 232, 178, 400, 320, 550, 480, 420, 470, 350])


# m = rows n = column in this case m =3 n =4



# Display the input data
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)


# w is a vector with n elements and b is a scalar parameter
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


# single prediction element by element without vectorization

def predict_single_loop(x, w, b):
    n = x.shape[0]  # Number of features in x
    p = 0           # Initialize prediction to 0
    for i in range(n):  # Loop through each feature
        p_i = x[i] * w[i]  # Compute contribution of feature i
        p = p + p_i        # Add contribution to the prediction
    p = p + b              # Add the bias term
    return p               # Return the final prediction



# get a row from the training data
#x_vec = X_train[0,:]
#print(f"x_vec shape {x_vec.shape}, x_velc value: {x_vec}")

# make a prediction
#f_wb = predict_single_loop(x_vec , w_init, b_init)
#print(f"prediction: {f_wb}")


# single prediction vector much faster
def predict(x, w, b):
    p = np.dot(x, w) + b
    return p


x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

f_wb = predict(x_vec, w_init, b_init)
print(f"prediction: {f_wb}")


# compute cost

def compute_cost(X, y, w,b):

    m = X.shape[0] # Number of training examples
    cost = 0.0  # Initialize cost to 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) +b  # Predicted value for example i
        cost = cost + (f_wb_i - y[i]) **2 # Add squared error to cost (scalar)
    cost = cost / (2 * m)  # Compute the average cost  (scalar)
    return cost



#compute and display

cost = compute_cost(X_train, y_train ,w_init ,b_init)
print(f"Cost at optimal w: {cost}")



# compute gradient with multiple variables

def compute_gradient(X,y , w,b):
    m,n = X.shape #(number of examples, number of features)

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i], w) +b) -y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db,  dj_dw



# compute and display the gradient


tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train , w_init , b_init)
print(f"dj_db at initial w,b: {tmp_dj_db}")
print(f"dj_dw at initial w,b: \n {tmp_dj_dw}")


# Gradiend Descent with multiple Variables

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing


# initialize parameters 
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient decent settings
iterations = 1000
alpha = 5.0e-7
# run gradient decent
w_final , b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f}, {w_final}")

m,_ = X_train.shape
for i in range(m):
     print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")





# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()



def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) /sigma

    return (X_norm, sigma, mu)


mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma      


# Define feature names for labeling axes (optional, replace as needed)
X_features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]

# Plot distributions
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
# Raw data
ax[0].scatter(X_train[:, 0], X_train[:, 1])
ax[0].set_title("Raw Data")
ax[0].set_xlabel(X_features[0])
ax[0].set_ylabel(X_features[1])
ax[0].axis('equal')

# Mean-subtracted data
ax[1].scatter(X_mean[:, 0], X_mean[:, 1])
ax[1].set_title("Mean-Subtracted Data")
ax[1].set_xlabel(X_features[0])
ax[1].set_ylabel(X_features[1])
ax[1].axis('equal')

# Z-Score normalized data
ax[2].scatter(X_norm[:, 0], X_norm[:, 1])
ax[2].set_title("Z-Score Normalized Data")
ax[2].set_xlabel(X_features[0])
ax[2].set_ylabel(X_features[1])
ax[2].axis('equal')

plt.tight_layout()
plt.show()
w_norm, b_norm, hist = gradient_descent(
    X_norm, y_train, np.zeros_like(w_init), 0.0, compute_cost, compute_gradient, 1.0e-1, 1000
)
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")


x_house_two = np.array([1400, 4, 2, 20])
x_house_two_norm = (x_house_two - X_mu) / X_sigma
print(x_house_norm)
x_house_two_predict  = np.dot(x_house_two_norm, w_norm) + b_norm
print(f" predicted price of a house with 1400 sqft, 4 bedrooms, 2 floor, 20 years old = ${x_house_two_predict*1000:0.0f}")


