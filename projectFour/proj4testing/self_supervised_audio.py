
"""
self_supervised_audio.py
Self-supervised training with a delay-line input from audio.

Two tasks:
A) Identity/Reconstruction: predict current sample x[t] from a 64-tap delay vector
B) 8-step-ahead prediction: predict x[t+8] with the same hidden size
   - First train output layer only (freeze hidden)
   - Then fine-tune full network

Also computes covariance eigenvalue spread of inputs from one epoch and
prints the condition number.

If no audio path is provided, generates a synthetic "music-like" signal.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from backprop_ode import TwoLayerNet, mse_loss, mse_grad

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

def load_audio_mono(path: str | None, fs_target: int = 16000, seconds: int = 20) -> np.ndarray:
    if path is None or (wavfile is None):
        # synthetic "music-like": sum of AM+FM sinusoids
        fs = fs_target
        t = np.arange(0, seconds*fs) / fs
        carrier1 = np.sin(2*np.pi*220*t + 0.3*np.sin(2*np.pi*3*t))
        carrier2 = 0.6*np.sin(2*np.pi*440*t + 0.2*np.sin(2*np.pi*2*t))
        bass = 0.5*np.sin(2*np.pi*55*t)
        sig = carrier1 + carrier2 + bass
        sig /= np.max(np.abs(sig)) + 1e-12
        return sig.astype(np.float32)
    # real audio
    fs, x = wavfile.read(path)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    # simple resample if needed (nearest neighbor)
    if fs != fs_target:
        r = fs_target / fs
        idx = (np.arange(int(len(x)*r)) / r).astype(int)
        idx = np.clip(idx, 0, len(x)-1)
        x = x[idx]
    x /= np.max(np.abs(x)) + 1e-12
    return x

def make_delay_matrix(x: np.ndarray, taps: int = 64, pred_ahead: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct column-major samples:
      X[:,k] = [x[k], x[k-1], ..., x[k-(taps-1)]]
      y[k]   = x[k+pred_ahead]
    We drop the first (taps-1) and last pred_ahead samples to align shapes.
    """
    N = len(x)
    K = N - taps - pred_ahead + 1
    X = np.zeros((taps, K), dtype=np.float32)
    for i in range(taps):
        X[i, :] = x[i : i+K]
    # reverse rows so X[0] is most recent sample if desired
    X = np.flipud(X)
    y = x[taps-1+pred_ahead : taps-1+pred_ahead+K][None, :]
    return X, y

def batch_iter(N, batch_size, rng):
    idx = np.arange(N)
    rng.shuffle(idx)
    for s in range(0, N, batch_size):
        b = idx[s:s+batch_size]
        yield b

def train_identity(x, n_hidden=16, taps=64, epochs=30, eta=0.01, freeze_hidden=False, seed=0, plot_path="/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/claude/identity_loss.png"):
    X, y = make_delay_matrix(x, taps=taps, pred_ahead=0)  # identity: predict current sample
    net = TwoLayerNet(
        n_in=taps, n_hidden=n_hidden, n_out=1,
        hidden_activation="sigmoid",
        output_activation="linear",
        gain_hidden=2.0,
        seed=seed
    )
    rng = np.random.default_rng(seed)
    losses = []
    for ep in range(epochs):
        for b in batch_iter(X.shape[1], 64, rng):
            Xb = X[:, b]
            yb = y[:, b]
            yhat, cache = net.forward(Xb)
            dL_dYh = mse_grad(yhat, yb)
            grads = net.backprop(cache, dL_dYh)
            if freeze_hidden:
                grads["W1"][:] = 0.0
                grads["b1"][:] = 0.0
            net.ode_step(grads, eta=eta, dt_w=1.0)
        yh, _ = net.forward(X)
        losses.append(mse_loss(yh, y))

    # plot loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Identity/Reconstruction Loss")
    plt.tight_layout()
    plt.savefig(plot_path)
    return net, X, y, losses

def train_predict8(x, base_net: TwoLayerNet, n_hidden=16, taps=64, epochs_head=3, epochs_finetune=5, eta_head=0.01, eta_full=0.005, seed=0, plot_path="/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/claude/predict8_loss.png"):
    X, y = make_delay_matrix(x, taps=taps, pred_ahead=8)
    # copy net
    import copy
    net = copy.deepcopy(base_net)

    rng = np.random.default_rng(seed)
    losses = []

    # phase 1: train output layer only
    for ep in range(epochs_head):
        for b in batch_iter(X.shape[1], 64, rng):
            Xb = X[:, b]
            yb = y[:, b]
            yhat, cache = net.forward(Xb)
            dL_dYh = mse_grad(yhat, yb)
            grads = net.backprop(cache, dL_dYh)
            # freeze hidden
            grads["W1"][:] = 0.0
            grads["b1"][:] = 0.0
            net.ode_step(grads, eta=eta_head, dt_w=1.0)
        yh, _ = net.forward(X)
        losses.append(mse_loss(yh, y))

    # phase 2: fine-tune full network
    for ep in range(epochs_finetune):
        for b in batch_iter(X.shape[1], 64, rng):
            Xb = X[:, b]
            yb = y[:, b]
            yhat, cache = net.forward(Xb)
            dL_dYh = mse_grad(yhat, yb)
            grads = net.backprop(cache, dL_dYh)
            net.ode_step(grads, eta=eta_full, dt_w=1.0)
        yh, _ = net.forward(X)
        losses.append(mse_loss(yh, y))

    # plot loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("8-Step-Ahead Prediction Loss (head-only -> full fine-tune)")
    plt.tight_layout()
    plt.savefig(plot_path)
    return net, X, y, losses

def eigen_spread_for_epoch(X_epoch: np.ndarray):
    """
    X_epoch: (taps, K) column-major samples
    Returns eigenvalues (sorted desc) and condition number.
    """
    # covariance across samples
    # Note: use row=feature, col=samples convention
    Xc = X_epoch - X_epoch.mean(axis=1, keepdims=True)
    C = (Xc @ Xc.T) / Xc.shape[1]
    # numerical safeguard
    C = 0.5*(C + C.T)
    w = np.linalg.eigvalsh(C)
    w = np.sort(w)[::-1]
    cond = (w[0] / (w[-1] + 1e-12)) if w[-1] > 0 else np.inf
    return w, cond

def run_all(audio_path: str | None = None, taps: int = 64):
    x = load_audio_mono(audio_path, fs_target=16000, seconds=5)
    # Identity training
    net_id, X_id, y_id, loss_id = train_identity(x, n_hidden=16, taps=taps, epochs=5, eta=0.01, seed=1, plot_path="/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/claude/identity_loss.png")
    # Eigen spread (on identity epoch design matrix)
    evals, cond = eigen_spread_for_epoch(X_id)
    print(f"Top-5 eigenvalues: {evals[:5]}")
    print(f"Eigenvalue condition number: {cond:.2e} (larger => slower LMS-style convergence)")

    # Predict +8 training
    net_p8, X_p8, y_p8, loss_p8 = train_predict8(x, base_net=net_id, n_hidden=16, taps=taps, epochs_head=3, epochs_finetune=5, eta_head=0.01, eta_full=0.005, seed=2, plot_path="/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/claude/predict8_loss.png")

    # Save small report
    rep = Path("/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/claude/self_supervised_report.txt")
    with rep.open("w") as f:
        f.write("Self-Supervised Audio Training Report\n")
        f.write("-------------------------------------\n")
        f.write(f"Samples used: {len(x)}\n")
        f.write(f"Taps: {taps}\n")
        f.write(f"Identity final MSE: {loss_id[-1]:.6e}\n")
        f.write(f"Predict+8 final MSE: {loss_p8[-1]:.6e}\n")
        f.write(f"Eigen cond #: {cond:.6e}\n")
        f.write("Note: Larger eigenvalue spreads (higher condition number) imply slower convergence for gradient/LMS-type methods; step sizes must be chosen below 2/λ_max for stability, and effective progress along small-eigen directions becomes slow when κ=λ_max/λ_min is large.\n")
    print(f"Wrote {rep}")

if __name__ == "__main__":
    run_all(audio_path="/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/withWords.wav", taps=64)
