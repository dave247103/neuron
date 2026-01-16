import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap

from main import Group, Neuron, _activation_np, _default_threshold, train_test_split, f1_score_binary


class NeuronGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neuron")
        self.geometry("1280x720")

        self.X0 = None
        self.X1 = None
        self.X = None
        self.D = None
        self.neuron = None

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        # Data generation controls
        ttk.Label(left, text="Data generation").pack(anchor="w", pady=(0, 5))

        self.modes0_var = tk.IntVar(value=3)
        self.modes1_var = tk.IntVar(value=3)
        self.spm_var = tk.IntVar(value=100)

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Modes (class 0):", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.modes0_var, width=10).pack(side=tk.LEFT)

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Modes (class 1):", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.modes1_var, width=10).pack(side=tk.LEFT)

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Samples/mode:", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.spm_var, width=10).pack(side=tk.LEFT)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10)

        # Training controls
        ttk.Label(left, text="Training").pack(anchor="w", pady=(0, 5))

        self.activation_var = tk.StringVar(value="sigmoid")
        self.beta_var = tk.DoubleVar(value=1.0)
        self.eta_var = tk.DoubleVar(value=0.05)
        self.epochs_var = tk.IntVar(value=30)

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Activation:", width=16).pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.activation_var,
            values=["heaviside", "sigmoid", "sin", "tanh", "relu", "lrelu", "sign"],
            state="readonly",
            width=12
        ).pack(side=tk.LEFT)

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Beta (sigmoid):", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.beta_var, width=10).pack(side=tk.LEFT)

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Learning rate:", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.eta_var, width=10).pack(side=tk.LEFT)

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Epochs:", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.epochs_var, width=10).pack(side=tk.LEFT)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10)

        # Actions
        ttk.Button(left, text="Generate data", command=self.on_generate).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Train neuron", command=self.on_train).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Clear plot", command=self.on_clear).pack(fill=tk.X, pady=3)

        self.status = ttk.Label(left, text="Status: idle", wraplength=240)
        self.status.pack(fill=tk.X, pady=(10, 0))

    def _build_plot(self):
        right = ttk.Frame(self, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._redraw(show_boundary=False)

    def on_clear(self):
        self.X0 = self.X1 = self.X = self.D = None
        self.neuron = None
        self.status.config(text="Status: cleared")
        self._redraw(show_boundary=False)

    def on_generate(self):
        try:
            n0 = int(self.modes0_var.get())
            n1 = int(self.modes1_var.get())
            spm = int(self.spm_var.get())
            if n0 <= 0 or n1 <= 0 or spm <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid input", "Modes and samples/mode must be positive integers.")
            return

        class_0 = Group(label=0, n_modes=n0, samples_per_mode=spm)
        class_1 = Group(label=1, n_modes=n1, samples_per_mode=spm)

        self.X0 = np.asarray(class_0.generate_samples(), dtype=float)
        self.X1 = np.asarray(class_1.generate_samples(), dtype=float)
        self.X = np.vstack([self.X0, self.X1])
        self.D = np.asarray([0] * len(self.X0) + [1] * len(self.X1), dtype=float)

        self.neuron = None  # force re-init on train
        self.status.config(text=f"Status: generated {len(self.X)} samples ({len(self.X0)} + {len(self.X1)})")
        self._redraw(show_boundary=False)

    def on_train(self):
        if self.X is None or self.D is None:
            messagebox.showwarning("No data", "Generate data first.")
            return

        act = self.activation_var.get().strip()
        beta = float(self.beta_var.get())
        eta = float(self.eta_var.get())
        epochs = int(self.epochs_var.get())

        if eta <= 0 or epochs <= 0:
            messagebox.showerror("Invalid input", "Learning rate and epochs must be positive.")
            return

        # Split
        X_train, D_train, X_test, D_test = train_test_split(self.X, self.D, test_ratio=0.2, seed=0)

        # Create neuron
        self.neuron = Neuron(n_inputs=2, activation=act, beta=beta, seed=0)

        # Train on train subset only
        self.neuron.train(X_train, D_train, eta=eta, epochs=epochs)

        # Evaluate on both
        yhat_train = np.array([self.neuron.predict(x) for x in X_train], dtype=int)
        f1_train = f1_score_binary(D_train, yhat_train, positive=1)

        yhat_test = np.array([self.neuron.predict(x) for x in X_test], dtype=int)
        f1_test = f1_score_binary(D_test, yhat_test, positive=1)

        self.status.config(text=f"Status: trained ({act}), train f1={f1_train:.3f}, test f1={f1_test:.3f}")
        self._redraw(show_boundary=True)


    def _redraw(self, show_boundary: bool):
        self.ax.clear()

        if self.X0 is None or self.X1 is None:
            self.ax.set_title("No data")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.canvas.draw()
            return

        X = np.vstack([self.X0, self.X1])
        pad = 0.5
        x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
        y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

        if show_boundary and self.neuron is not None:
            grid = 300
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, grid),
                np.linspace(y_min, y_max, grid)
            )

            # Correct bias handling: x_star = [x, y, -1]
            w0, w1, w_bias = self.neuron.w
            s = w0 * xx + w1 * yy - w_bias

            act = self.neuron.activation_name
            threshold = _default_threshold(act)

            # Half-plane cases: boundary is s=0
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
                # General case: boundary is f(s)=threshold
                y = _activation_np(act, s, self.neuron.beta)
                Z = (y >= threshold).astype(int)
                boundary_field = y - threshold


            cmap_bg = ListedColormap(["#dbe9ff", "#ffd6d6"])
            self.ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap=cmap_bg, alpha=0.6)
            self.ax.contour(xx, yy, boundary_field, levels=[0.0], colors="k", linewidths=1.5)

        self.ax.scatter(self.X0[:, 0], self.X0[:, 1], color="blue", label="Class 0", alpha=0.6)
        self.ax.scatter(self.X1[:, 0], self.X1[:, 1], color="red", label="Class 1", alpha=0.6)

        self.ax.set_title("Data + decision regions" if show_boundary and self.neuron else "Data distribution")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.legend(loc="best")

        self.canvas.draw()


if __name__ == "__main__":
    app = NeuronGUI()
    app.mainloop()
