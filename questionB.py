# Differntition

from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import sympy as sp

# Define the function
x = sp.Symbol('x')
f = x**3 + 2*x**2 - 5*x + 1

# Differentiate the function
derivative = sp.diff(f, x)

print("Function:", f)
print("Derivative:", derivative)

# Numerical intergration

# Define the function


def f(x):
    return x**2 + 2*x


# Integrate the function from 0 to 5
result, error = quad(f, 0, 5)

print("Integral value:", result)
print("Absolute error estimate:", error)

# Curve Fitting

# Generate sample data
x = np.linspace(0, 10, 50)
y = 2*x + 3 + np.random.normal(0, 2, 50)

# Define the curve fitting function


def linear_func(x, a, b):
    return a*x + b


# Perform curve fitting
popt, pcov = curve_fit(linear_func, x, y)

# Print the fitted parameters
print("Fitted parameters: a =", popt[0], ", b =", popt[1])

# Plot the data and fitted curve
plt.scatter(x, y, label='Data')
plt.plot(x, linear_func(x, *popt), color='r', label='Fitted Curve')
plt.legend()
plt.show()

# Linear Regression

# Generate sample data
x = np.array([[1, 2], [1, 4], [2, 2], [2, 4], [3, 5]])
y = np.array([3, 7, 5, 11, 13])

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(x, y)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Make predictions
predictions = model.predict(x)

# Plot the data and regression line
plt.scatter(x[:, 0], y, label='Data')
plt.plot(x[:, 0], predictions, color='r', label='Regression Line')
plt.legend()
plt.show()

# Spine interpolation


# Generate sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 5, 8, 5, 3])

# Create a spline interpolation object
spline = interp1d(x, y, kind='cubic')

# Generate new x values for interpolation
x_new = np.linspace(1, 5, 100)

# Perform spline interpolation
y_new = spline(x_new)

# Plot the original data and interpolated curve
plt.scatter(x, y, label='Data')
plt.plot(x_new, y_new, color='r', label='Spline Interpolation')
plt.legend()
plt.show()
