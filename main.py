import random
import matplotlib.pyplot as plt

class Mode:
    def __init__(self, mean_x: float, mean_y: float, variance_x: float, variance_y: float):
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.variance_x = variance_x
        self.variance_y = variance_y
    
    def sample(self, n: int):
        return [(random.gauss(self.mean_x, self.variance_x**0.5), random.gauss(self.mean_y, self.variance_y**0.5)) for _ in range(n)]
    
class Group:
    def __init__(self, label: int, n_modes: int, samples_per_mode: int, mean_range=(-1.0, 1.0), variance_range=(0.5, 2.0)):
        self.label = label
        self.modes = [Mode(random.uniform(*mean_range), random.uniform(*mean_range), random.uniform(*variance_range), random.uniform(*variance_range)) for _ in range(n_modes)]
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

class_0 = Group(label=0, n_modes=3, samples_per_mode=100)
class_1 = Group(label=1, n_modes=3, samples_per_mode=100)

X_0 = class_0.generate_samples()
X_1 = class_1.generate_samples()
plot_data(X_0, X_1)