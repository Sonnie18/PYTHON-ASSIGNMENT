import numpy as np
import matplotlib.pyplot as plt


def trapezoidal_rule(func, a, b, n):
    """
    Approximate the integral of func from a to b using the trapezoidal rule.
    
    Parameters:
    func : function
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    n : int
        The number of trapezoids.
    
    Returns:
    float
        The approximate integral.
    """
    # Calculate the width of each trapezoid
    h = (b - a) / n

    # Calculate the x values at which to evaluate the function
    x = np.linspace(a, b, n + 1)

    # Calculate the function values at the x values
    y = func(x)

    # Calculate the area using the trapezoidal rule
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[n])

    return integral, x, y

# Define the function to integrate


def my_function(x):
    return np.sin(x)  # Example function: sin(x)


# Set the limits of integration and number of trapezoids
a = 0  # Lower limit
b = np.pi  # Upper limit
n = 10  # Number of trapezoids

# Calculate the integral
integral, x, y = trapezoidal_rule(my_function, a, b, n)

# Print the result
print(f"Approximate integral of sin(x) from {a} to {
      b} using {n} trapezoids: {integral:.4f}")

# Plotting the function and trapezoids
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b', label='f(x) = sin(x)', linewidth=2)
plt.fill_between(x, y, color='lightgray', alpha=0.5, label='Trapezoids')
plt.title('Trapezoidal Rule Integration')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()
