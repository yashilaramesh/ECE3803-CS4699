
"""
backprop_ode.py
A small, parameterized two-layer backprop network with an ODE-style weight update
and a simple LMS module. Designed to be reusable for classification or regression.

- TwoLayerNet:
    * Arbitrary input/hidden/output sizes
    * Sigmoid hidden with configurable gain `g`
    * Linear or sigmoid output
    * ODE-style training loop: dW/dt = -eta * dL/dW (integrated with Euler updates)
- LMSFilter:
    * Classic LMS adaptive linear filter (one layer), useful as a warm-up

This file has no external dependencies beyond numpy and (optionally) matplotlib for demos.
"""
from __future__ import annotations
import numpy as np

# ---------------------- activations ----------------------
def sigmoid(z: np.ndarray, g: float = 1.0) -> np.ndarray:
    # numerically-stable sigmoid with adjustable gain
    # σ_g(z) = 1/(1+exp(-g z)) mapped to (-1,1) if desired by postprocessing
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-g * z))

def dsigmoid_from_activation(a: np.ndarray, g: float = 1.0) -> np.ndarray:
    # derivative using activation value: σ'(z) = g * a * (1-a)
    return g * a * (1.0 - a)

def tanh(z: np.ndarray, g: float = 1.0) -> np.ndarray:
    z = np.clip(g * z, -20.0, 20.0)
    return np.tanh(z)

def dtanh_from_activation(a: np.ndarray, g: float = 1.0) -> np.ndarray:
    return g * (1.0 - a**2)

# ---------------------- losses ---------------------------
def mse_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y_hat - y)**2))

def mse_grad(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (2.0 / y.shape[0]) * (y_hat - y)

# ---------------------- network --------------------------
class TwoLayerNet:
    """
    Input -> Hidden (sigmoid/tanh) -> Output (linear or sigmoid)
    ODE-style backprop: dθ/dt = -η * ∂L/∂θ, Euler integration with step dt_w.
    """
    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        n_out: int,
        hidden_activation: str = "sigmoid",   # "sigmoid" or "tanh"
        output_activation: str = "linear",    # "linear" or "sigmoid"
        gain_hidden: float = 2.0,
        gain_output: float = 1.0,
        weight_scale: float = 0.5,
        seed: int | None = 0,
    ):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.g_h = gain_hidden
        self.g_o = gain_output

        self.W1 = rng.normal(0.0, weight_scale/np.sqrt(n_in), size=(n_hidden, n_in))
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = rng.normal(0.0, weight_scale/np.sqrt(n_hidden), size=(n_out, n_hidden))
        self.b2 = np.zeros((n_out, 1))

    # ---------- forward ----------
    def _act(self, z, which, g):
        if which == "sigmoid":
            return sigmoid(z, g)
        elif which == "tanh":
            return tanh(z, g)
        elif which == "linear":
            return z
        else:
            raise ValueError(f"Unknown activation: {which}")

    def _dact(self, a, which, g):
        if which == "sigmoid":
            return dsigmoid_from_activation(a, g)
        elif which == "tanh":
            return dtanh_from_activation(a, g)
        elif which == "linear":
            return np.ones_like(a)
        else:
            raise ValueError(f"Unknown activation: {which}")

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        X: shape (n_in, B) with column-major samples
        Returns (y_hat, cache)
        """
        Z1 = self.W1 @ X + self.b1
        H  = self._act(Z1, self.hidden_activation, self.g_h)
        Z2 = self.W2 @ H + self.b2
        Yh = self._act(Z2, self.output_activation, self.g_o)
        cache = {"X": X, "Z1": Z1, "H": H, "Z2": Z2, "Yh": Yh}
        return Yh, cache

    # ---------- backprop (batch) ----------
    def backprop(self, cache: dict, dL_dYh: np.ndarray) -> dict:
        H = cache["H"]; X = cache["X"]; Yh = cache["Yh"]
        Z2 = cache["Z2"]; Z1 = cache["Z1"]

        dYh_dZ2 = self._dact(Yh, self.output_activation, self.g_o)
        dL_dZ2  = dL_dYh * dYh_dZ2                         # (n_out, B)

        dL_dW2 = (dL_dZ2 @ H.T) / X.shape[1]
        dL_db2 = np.mean(dL_dZ2, axis=1, keepdims=True)

        dZ2_dH = self.W2.T
        dL_dH  = dZ2_dH @ dL_dZ2                            # (n_hidden, B)

        dH_dZ1 = self._dact(cache["H"], self.hidden_activation, self.g_h)
        dL_dZ1 = dL_dH * dH_dZ1

        dL_dW1 = (dL_dZ1 @ X.T) / X.shape[1]
        dL_db1 = np.mean(dL_dZ1, axis=1, keepdims=True)

        grads = {"W1": dL_dW1, "b1": dL_db1, "W2": dL_dW2, "b2": dL_db2}
        return grads

    # ---------- ODE/Euler update ----------
    def ode_step(self, grads: dict, eta: float, dt_w: float = 1.0):
        # dθ/dt = -eta * grad  => θ_{t+Δ} = θ_t - (eta * dt_w) * grad
        step = eta * dt_w
        self.W1 -= step * grads["W1"]
        self.b1 -= step * grads["b1"]
        self.W2 -= step * grads["W2"]
        self.b2 -= step * grads["b2"]

    # ---------- training ----------
    def fit(
        self,
        X: np.ndarray, Y: np.ndarray,
        epochs: int = 2000,
        batch_size: int = 64,
        eta: float = 0.5,
        dt_w: float = 1.0,
        loss_fn=mse_loss,
        loss_grad=mse_grad,
        shuffle: bool = True,
        verbose_every: int = 0,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
    ) -> dict:
        """
        Column-major data:
          X: (n_in, N), Y: (n_out, N)
        """
        N = X.shape[1]
        idx = np.arange(N)
        history = {"loss": [], "val_loss": []}

        for ep in range(epochs):
            if shuffle:
                np.random.shuffle(idx)
            # mini-batches
            for start in range(0, N, batch_size):
                batch = idx[start:start+batch_size]
                Xb = X[:, batch]
                Yb = Y[:, batch]
                Yh, cache = self.forward(Xb)
                dL_dYh = loss_grad(Yh, Yb)
                grads = self.backprop(cache, dL_dYh)
                self.ode_step(grads, eta=eta, dt_w=dt_w)

            # log losses
            Yh_full, _ = self.forward(X)
            train_loss = loss_fn(Yh_full, Y)
            history["loss"].append(train_loss)
            if X_val is not None and Y_val is not None:
                Yh_val, _ = self.forward(X_val)
                val_loss = loss_fn(Yh_val, Y_val)
                history["val_loss"].append(val_loss)

            if verbose_every and (ep % verbose_every == 0):
                if X_val is not None:
                    print(f"epoch {ep:4d} | loss={train_loss:.4e} | val={val_loss:.4e}")
                else:
                    print(f"epoch {ep:4d} | loss={train_loss:.4e}")

        return history

# ---------------------- LMS (warm-up) ---------------------
class LMSFilter:
    """
    Classic LMS: y = w^T x ; e = d - y ; w <- w + mu * e * x
    Here we also show an ODE view: dw/dt = -∂(e^2)/∂w = 2 e (-x)
    With Euler update: w <- w + (mu * dt_w) * e * x
    """
    def __init__(self, n_in: int, seed: int | None = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.1, size=(n_in,))

    def step(self, x: np.ndarray, d: float, mu: float = 0.01, dt_w: float = 1.0) -> float:
        y = float(self.w @ x)
        e = d - y
        self.w += (mu * dt_w) * e * x
        return e

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w
