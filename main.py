import math

def gaussian(x, mu, sigma):
    """Compute the Gaussian function value at x.

    Args:
        x (float): The input value.
        mu (float): The mean of the Gaussian.
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        float: The value of the Gaussian function at x.
    """
    return 1 / math.sqrt(2 * math.pi * sigma ** 2) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

