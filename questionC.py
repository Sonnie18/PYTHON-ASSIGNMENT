import numpy as np


def f(x, y):
    """Compute the value of the function f(x, y)."""
    return x**2 + y**2 - x*y + x - y + 1


def gradient_f(x, y):
    """Compute the gradient of the function f(x, y)."""
    df_dx = 2*x - y + 1  # Partial derivative with respect to x
    df_dy = 2*y - x - 1  # Partial derivative with respect to y
    return np.array([df_dx, df_dy])


def gradient_descent(initial_guess, learning_rate=0.1, max_iterations=100):
    """Perform gradient descent to minimize the function f(x, y)."""
    x, y = initial_guess
    for iteration in range(max_iterations):
        grad = gradient_f(x, y)
        x -= learning_rate * grad[0]  # Update x
        y -= learning_rate * grad[1]  # Update y

        # Print the current state
        print(f"Iteration {
              iteration + 1}: x = {x:.4f}, y = {y:.4f}, f(x, y) = {f(x, y):.4f}")

    return x, y


# Example usage
initial_guess = (0, 0)  # Starting point (X0, Y0)
learning_rate = 0.1
max_iterations = 100

optimal_x, optimal_y = gradient_descent(
    initial_guess, learning_rate, max_iterations)

print(f"\nOptimal point: x = {optimal_x:.4f}, y = {optimal_y:.4f}")
print(f"Minimum value of f(x, y) = {f(optimal_x, optimal_y):.4f}")
