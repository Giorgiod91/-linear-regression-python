import numpy as np
import matplotlib.pyplot as plt

#dates for the nerfs 
x_train  = np.array([17.9,9.10,])
#nerf in %
y_train = np.array([6,5])

#finding the value w(slope) and the b(intercept)
def find_values(x,y):

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    #variance and covariance
    var_x = np.sum((x - x_mean) ** 2)
    cov_xy = np.sum((x - x_mean) * (y - y_mean))

    w = cov_xy / var_x
    b = y_mean - w * x_mean

    return w,b

w, b= find_values(x_train,y_train)

#model prediction from the costFunction i learned
def nerf_prediction(x,w,b):
    # Number of training examples
    return w * x + b


predicted_nerfs = nerf_prediction(x_train,w,b)

# Convert dates to strings to make them readable on the x-axis
x_labels = ['17.9', '9.10']


#plot

plt.plot(x_labels, predicted_nerfs, c="b", label="lol")
plt.scatter(x_labels, y_train, marker='x', color='r', label='Actual Values')
plt.title('Nerf Predictions Based on Dates')
plt.xlabel('Dates (day.month)')
plt.ylabel('Nerf Percentage')
plt.legend()
plt.show()


#input dates for prediction

new_dates = np.array([20.10, 24.11,18.12])

#same here convert to strings
new_dates_converted = ["20.10", "24.11", "18.12"]

predicted_nerfs = nerf_prediction(new_dates,w,b)

#plot predictions for dates
plt.plot(new_dates_converted, predicted_nerfs, c="b", label="lol2")
plt.scatter(new_dates_converted, predicted_nerfs, marker="x", color="r", label="avtual")
plt.title("Are we getting nerfed?")
plt.xlabel('Dates (day.month)')
plt.ylabel('Nerf Percentage')
plt.legend()
plt.show()




# Print out the predictions for the new dates
for date, nerf in zip(new_dates_converted, predicted_nerfs):
    print(f"Predicted nerf percentage for {date}: {nerf:.2f}%")





