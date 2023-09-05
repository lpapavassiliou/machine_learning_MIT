import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x, y):
    X = np.column_stack((x, np.ones_like(x)))
    coefficients, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    slope, intercept = coefficients
    return slope, intercept

# Input data
x_numpy = np.array([0.233, 1.347, 2.53, 3.653, 4.768]).reshape((-1,1))
y_numpy = np.array([-20, -10, 0, 10, 20])

# Apply linear regression
slope, intercept = linear_regression(x_numpy, y_numpy)
print("Slope:", slope)
print("Intercept:", intercept)

# Create a line based on the slope and intercept
x_line = np.linspace(min(x_numpy), max(x_numpy), 100)
y_line = slope * x_line + intercept

# Plot the data points and the regression line
plt.scatter(x_numpy, y_numpy, label='Data')
plt.plot(x_line, y_line, color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
