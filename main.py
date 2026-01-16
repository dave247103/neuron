import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# data generation
class Mode:
    def __init__(self, mean_x: float, mean_y: float, variance_x: float, variance_y: float):
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.variance_x = variance_x
        self.variance_y = variance_y
    
    def sample(self, n: int):
        return [(random.gauss(self.mean_x, self.variance_x**0.5),
                 random.gauss(self.mean_y, self.variance_y**0.5)) for _ in range(n)]    
class Group:
    def __init__(self, label: int, n_modes: int, samples_per_mode: int,
                  mean_range=(-1.0, 1.0), variance_range=(0.5, 2.0)):
        self.label = label
        self.modes = [
            Mode(random.uniform(*mean_range),
                 random.uniform(*mean_range),
                 random.uniform(*variance_range),
                 random.uniform(*variance_range))
            for _ in range(n_modes)]
        self.samples_per_mode = samples_per_mode

    def generate_samples(self):
        return [sample for mode in self.modes for sample in mode.sample(self.samples_per_mode)]

def plot_data(X_0, X_1):
    plt.scatter(*zip(*X_0), color='blue', label='Class 0', alpha=0.5)
    plt.scatter(*zip(*X_1), color='red', label='Class 1', alpha=0.5)
    plt.legend()
    plt.title('Data Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# activation functions
def heaviside(s: float) -> float:
    return 1.0 if s >= 0.0 else 0.0

def heaviside_derivative(s: float) -> float:
    return 1.0

def sigmoid(s: float, beta: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-beta * s))

def sigmoid_derivative(s: float, beta: float = 1.0) -> float:
    return beta * sigmoid(s, beta) * (1.0 - sigmoid(s, beta))

def sinus(s: float) -> float:
    return math.sin(s)

def sinus_derivative(s: float) -> float:
    return math.cos(s)

def tanh_act(s: float) -> float:
    return math.tanh(s)

def tanh_derivative(s: float) -> float:
    return 1.0 - math.tanh(s)**2

def sign(s: float) -> float:
    return 1.0 if s > 0.0 else -1.0 if s < 0.0 else 0.0

def sign_derivative(s: float) -> float:
    return 1.0

def ReLU(s: float) -> float:
    return max(0.0, s)

def ReLU_derivative(s: float) -> float:
    return heaviside(s)

def leaky_ReLU(s: float) -> float:
    return s if s > 0.0 else 0.01 * s

def leaky_ReLU_derivative(s: float) -> float:
    return 1.0 if s > 0.0 else 0.01

ACTIVATIONS = {
    'heaviside': (heaviside, heaviside_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'sin': (sinus, sinus_derivative),
    'tanh': (tanh_act, tanh_derivative),
    'relu': (ReLU, ReLU_derivative),
    'lrelu': (leaky_ReLU, leaky_ReLU_derivative),
    'sign': (sign, sign_derivative),
}


# neuron
class Neuron:
    def __init__(self, n_inputs: int, activation: str = 'sigmoid', beta: float = 1.0, seed: int | None = None):
        self.activation_name = activation
        self.beta = beta
        self.w = np.random.default_rng(seed).normal(loc=0.0, scale = 0.1, size=n_inputs+1)

    def phi(self, s: float) -> float:
        f, _ = ACTIVATIONS[self.activation_name]
        return f(s, self.beta) if self.activation_name == 'sigmoid' else f(s)
    
    def phi_derivative(self, s: float) -> float:
        _, f_derivative = ACTIVATIONS[self.activation_name]
        return f_derivative(s, self.beta) if self.activation_name == 'sigmoid' else f_derivative(s)
    
    def forward_pass(self, x):
        x = np.asarray(x, dtype=float)
        x_star = np.append(x, -1.0)
        s = float(self.w @ x_star)
        y = float(self.phi(s))
        return y, s, x_star

    # returns predicted class for input x
    def predict(self, x, threshold: float | None = None) -> int:
        y, s, _ = self.forward_pass(x)

        if threshold is None:
            threshold = {
                "heaviside": 0.5,
                "sigmoid": 0.5,
                "tanh": 0.0,
                "sin": 0.0,
                "relu": 0.0,
                "lrelu": 0.0,
                "sign": 0.5,
            }[self.activation_name]

        if self.activation_name in ("sign", "relu"):
            return 1 if s > 0.0 else 0

        return 1 if y >= threshold else 0
 

    # trains one epoch and returns MSE
    def train_step(self, x, d: float, eta: float = 0.01):
        y, s, x_star = self.forward_pass(x)
        error = d - y
        delta_w = eta * error * self.phi_derivative(s) * x_star
        self.w += delta_w
        return error**2
    
    # trains multiple epochs
    def train(self, X, D, eta: float = 0.01, epochs: int = 100):
        for _ in range(epochs):
            idx = np.random.permutation(len(X)) # shuffle data each epoch
            for i in idx:
                self.train_step(X[i], D[i], eta)

def _activation_np(name: str, s: np.ndarray, beta: float = 1.0) -> np.ndarray:
    if name == "heaviside":
        return (s >= 0.0).astype(float)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-beta * s))
    if name == "sin":
        return np.sin(s)
    if name == "tanh":
        return np.tanh(s)
    if name == "sign":
        return np.sign(s)
    if name == "relu":
        return np.maximum(0.0, s)
    if name == "lrelu":
        return np.where(s > 0.0, s, 0.01 * s)
    raise ValueError(f"Unknown activation: {name}")

def _default_threshold(name: str) -> float:
    # Chosen so that (for monotone activations) the boundary is s = 0 (a half-plane).
    return {
        "heaviside": 0.5,
        "sigmoid": 0.5,
        "tanh": 0.0,
        "sin": 0.0,     # produces multiple stripes, not half-planes
        "relu": 0.0,
        "lrelu": 0.0,
        "sign": 0.5,    # class 1 only when output is +1
    }[name]

def plot_data_with_boundary(X_0, X_1, neuron, threshold: float | None = None,
                            grid: int = 400, pad: float = 0.5):
    X_0 = np.asarray(X_0, dtype=float)
    X_1 = np.asarray(X_1, dtype=float)
    X = np.vstack([X_0, X_1])

    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid),
                         np.linspace(y_min, y_max, grid))

    # weights: [w0, w1, w_bias] and you use x_star = [x, y, -1]
    w0, w1, w_bias = neuron.w
    s = w0 * xx + w1 * yy - w_bias  # NOTE the minus (matches x_star bias = -1)

    act = neuron.activation_name
    if threshold is None:
        threshold = _default_threshold(act)

    # Compute predicted regions
    if act == "heaviside":
        Z = (s >= 0.0).astype(int)
        boundary_field = s
    elif act == "sign":
        Z = (s > 0.0).astype(int)
        boundary_field = s
    elif act == "relu":
        Z = (s > 0.0).astype(int)
        boundary_field = s
    else:
        y = _activation_np(act, s, neuron.beta)
        Z = (y >= threshold).astype(int)
        boundary_field = y - threshold

    cmap_bg = ListedColormap(["#dbe9ff", "#ffd6d6"])  # light blue / light red
    plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap=cmap_bg, alpha=0.6)

    # Decision boundary (may be multiple curves for non-monotone activations like sin)
    plt.contour(xx, yy, boundary_field, levels=[0.0], colors="k", linewidths=1.5)

    plt.scatter(X_0[:, 0], X_0[:, 1], color="blue", label="Class 0", alpha=0.6)
    plt.scatter(X_1[:, 0], X_1[:, 1], color="red", label="Class 1", alpha=0.6)

    plt.legend()
    plt.title("Data distribution + decision regions")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


if __name__ == "__main__":
    # generate data
    class_0 = Group(label=0, n_modes=3, samples_per_mode=100)
    class_1 = Group(label=1, n_modes=3, samples_per_mode=100)
    X_0 = class_0.generate_samples()
    X_1 = class_1.generate_samples()

    X = np.array(X_0 + X_1, dtype=float)
    D = np.array([0]*len(X_0) + [1]*len(X_1), dtype=float)

    # train
    neuron = Neuron(n_inputs=2, activation="sigmoid", beta=1.0, seed=0)
    neuron.train(X, D, eta=0.05, epochs=30)

    print("Weights (including bias last):", neuron.w)
    print("Pred label for (0,0):", neuron.predict([0, 0]))

    plot_data_with_boundary(X_0, X_1, neuron)