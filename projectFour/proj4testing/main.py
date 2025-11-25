import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


def load_and_preprocess_audio(filepath):
    sr, audio = wavfile.read(filepath)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32), sr


def create_sliding_windows(audio, window_size=64, prediction_offset=0):
    N = len(audio) - window_size - prediction_offset
    X = np.empty((N, window_size), dtype=np.float32)
    if prediction_offset == 0:
        y = np.empty((N, window_size), dtype=np.float32)
    else:
        y = np.empty((N, prediction_offset), dtype=np.float32)

    for i in range(N):
        X[i] = audio[i:i+window_size]
        if prediction_offset == 0:
            y[i] = audio[i:i+window_size]
        else:
            y[i] = audio[i+window_size:i+window_size+prediction_offset]
    return X, y


def compute_covariance_analysis(X):
    Xc = X - np.mean(X, axis=0, keepdims=True)
    C = np.cov(Xc.T)
    w = np.linalg.eigvalsh(C)
    w = np.sort(w)[::-1]
    spread = w[0] / (w[-1] + 1e-10)
    return C, w, spread



def sigmoid(z, g=3.0):
    return g*np.tanh(z)

def dsigmoid_from_a(a, g=3.0):
    return g * a * (1.0 - a)

def forward(W1, W2, x, hidden_gain=3.0, use_output_sigmoid=False):
    """
    x: (D,) or (B,D) row-major
    Returns tuple (hidden, yhat) with shapes matching batchness.
    """
    batched = (x.ndim == 2)
    X = x if batched else x[None, :]          # (B,D)
    Z1 = X @ W1.T                              # (B,H)
    H  = sigmoid(Z1, g=hidden_gain)            # (B,H)
    Z2 = H @ W2.T                              # (B,O)
    Yh = sigmoid(Z2, g=hidden_gain) if use_output_sigmoid else Z2  # linear if False
    return (H if batched else H[0]), (Yh if batched else Yh[0])

def compute_dW(W1, W2, x, target, hidden, yhat, lr, tau_slow,
               hidden_gain=3.0, use_output_sigmoid=False):
    """
    Single-sample or batch gradient (MSE) -> dW1, dW2 and MSE.
    Shapes:
      x: (D,) or (B,D)
      target: (O,) or (B,O)
      hidden: (H,) or (B,H)
      yhat: (O,) or (B,O)
    """
    output_error = target - yhat
    output_delta = output_error *dsigmoid_from_a(hidden)
    hidden_error = W2.T@output_delta
    hidden_delta = hidden_error * dsigmoid_from_a(hidden)
    dW2_dt = (lr/tau_slow) * np.outer(output_delta, hidden)
    dW1_dt = (lr/tau_slow) * np.outer(hidden_delta,x)
    return np.mean(output_error**2)

def euler_step(W1, W2, dW1, dW2, dt=1.0):
    """Euler integrate weights."""
    W1 = W1 + dW1 * dt
    W2 = W2 + dW2 * dt
    return W1, W2

def rk4_step(W1, W2, x, target, lr, tau_slow, dt,
             hidden_gain=3.0, use_output_sigmoid=False):
    """
    RK4 on weights with current (x, target). Mirrors the class logic.
    """
    W1_0, W2_0 = W1.copy(), W2.copy()

    # k1
    h, y = forward(W1, W2, x, hidden_gain, use_output_sigmoid)
    dW1_1, dW2_1, _ = compute_dW(W1, W2, x, target, h, y, lr, tau_slow, hidden_gain, use_output_sigmoid)

    # k2
    W1 = W1_0 + 0.5*dt*dW1_1; W2 = W2_0 + 0.5*dt*dW2_1
    h, y = forward(W1, W2, x, hidden_gain, use_output_sigmoid)
    dW1_2, dW2_2, _ = compute_dW(W1, W2, x, target, h, y, lr, tau_slow, hidden_gain, use_output_sigmoid)

    # k3
    W1 = W1_0 + 0.5*dt*dW1_2; W2 = W2_0 + 0.5*dt*dW2_2
    h, y = forward(W1, W2, x, hidden_gain, use_output_sigmoid)
    dW1_3, dW2_3, _ = compute_dW(W1, W2, x, target, h, y, lr, tau_slow, hidden_gain, use_output_sigmoid)

    # k4
    W1 = W1_0 + dt*dW1_3; W2 = W2_0 + dt*dW2_3
    h, y = forward(W1, W2, x, hidden_gain, use_output_sigmoid)
    dW1_4, dW2_4, _ = compute_dW(W1, W2, x, target, h, y, lr, tau_slow, hidden_gain, use_output_sigmoid)

    W1 = W1_0 + (dt/6.0)*(dW1_1 + 2*dW1_2 + 2*dW1_3 + dW1_4)
    W2 = W2_0 + (dt/6.0)*(dW2_1 + 2*dW2_2 + 2*dW2_3 + dW2_4)

    # report error at end-of-step
    _, y_final = forward(W1, W2, x, hidden_gain, use_output_sigmoid)
    mse = float(np.mean((target - y_final)**2))
    return W1, W2, mse



def train_epoch_ode(X, Y, W1, W2, lr=1e-3, dt=1.0, use_rk4=False,
                    hidden_gain=3.0, use_output_sigmoid=False, tau_slow=100.0,
                    output_only=False):
    """
    One epoch over (X,Y) using ODE updates. Shuffles samples.
    If output_only=True, freezes W1 (hidden).
    """
    N = X.shape[0]
    idx = np.random.permutation(N)
    losses = []

    for i in idx:
        x = X[i]          # (D,)
        t = Y[i]          # (O,)

        h, y = forward(W1, W2, x, hidden_gain, use_output_sigmoid)

        if use_rk4:
            if output_only:
                dW1, dW2, _ = compute_dW(W1, W2, x, t, h, y, lr, tau_slow, hidden_gain, use_output_sigmoid)
                dW1[:] = 0.0
                W1, W2 = euler_step(W1, W2, dW1, dW2, dt)
                _, y2 = forward(W1, W2, x, hidden_gain, use_output_sigmoid)
                loss = float(np.mean((t - y2)**2))
            else:
                W1, W2, loss = rk4_step(W1, W2, x, t, lr, tau_slow, dt,
                                        hidden_gain, use_output_sigmoid)
        else:
            dW1, dW2, loss = compute_dW(W1, W2, x, t, h, y, lr, tau_slow, hidden_gain, use_output_sigmoid)
            if output_only:
                dW1[:] = 0.0
            W1, W2 = euler_step(W1, W2, dW1, dW2, dt)

        losses.append(loss)

    return W1, W2, float(np.mean(losses))


if __name__ == "__main__":

    audio_file = "/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/withWords.wav"
    audio_name = "4rDaDawgs"
    audio, sr = load_and_preprocess_audio(audio_file)
    print(f"Audio length: {len(audio)} samples ({len(audio)/sr:.2f} s) @ {sr} Hz")


    window_size = 64
    X_train, Y_train = create_sliding_windows(audio, window_size, prediction_offset=8)
    print(f"Training samples (repro): {len(X_train)}")

    C, evals, spread = compute_covariance_analysis(X_train)
    print(f"Eigenvalue spread (λmax/λmin): {spread:.2e}")
    print(f"Top-5 eigvals: {np.round(evals[:5], 6)}")
    print(f"Bottom-5 eigvals: {np.round(evals[-5:], 6)}")
    print(f"\nTimescale separation: τ_fast=1.0, τ_slow=100.0 (100x)")

    D = window_size
    H = 20
    O_repro = window_size

    rng = np.random.default_rng(0)
    W1_repro = rng.normal(0, 0.01, size=(H, D))
    W2_repro = rng.normal(0, 0.01, size=(O_repro, H))

    epochs_repro = 20
    lr = 0.001
    dt = 1.0
    hidden_gain = 3.0
    use_output_sigmoid = False
    tau_slow = 100.0

    errors_repro = []
    for ep in range(epochs_repro):
        W1_repro, W2_repro, avg_mse = train_epoch_ode(
            X_train, Y_train, W1_repro, W2_repro,
            lr=lr, dt=dt, use_rk4=False,
            hidden_gain=hidden_gain, use_output_sigmoid=use_output_sigmoid, tau_slow=tau_slow,
            output_only=False
        )
        errors_repro.append(avg_mse)
        print(f"Epoch {ep+1}/{epochs_repro}  MSE={avg_mse:.6f}")

    prediction_samples = 32
    X_pred, Y_pred = create_sliding_windows(audio, window_size, prediction_offset=prediction_samples)
    print(f"Training samples (pred): {len(X_pred)}")

    O_pred = prediction_samples
    W1_pred = W1_repro.copy()
    W2_pred = rng.normal(0, 0.01, size=(O_pred, H))

    epochs_stage1 = 10
    errors_pred_stage1 = []
    for ep in range(epochs_stage1):
        W1_pred, W2_pred, avg_mse = train_epoch_ode(
            X_pred, Y_pred, W1_pred, W2_pred,
            lr=0.001, dt=1.0, use_rk4=False,
            hidden_gain=hidden_gain, use_output_sigmoid=use_output_sigmoid, tau_slow=tau_slow,
            output_only=True
        )
        errors_pred_stage1.append(avg_mse)
        print(f"[Stage1] Epoch {ep+1}/{epochs_stage1}  MSE={avg_mse:.6f}")

    epochs_stage2 = 20
    errors_pred_stage2 = []
    for ep in range(epochs_stage2):
        W1_pred, W2_pred, avg_mse = train_epoch_ode(
            X_pred, Y_pred, W1_pred, W2_pred,
            lr=0.0005, dt=1.0, use_rk4=False,
            hidden_gain=hidden_gain, use_output_sigmoid=use_output_sigmoid, tau_slow=tau_slow,
            output_only=False
        )
        errors_pred_stage2.append(avg_mse)
        print(f"[Stage2] Epoch {ep+1}/{epochs_stage2}  MSE={avg_mse:.6f}")

    errors_pred_full = errors_pred_stage1 + errors_pred_stage2

    predicted_audio = []
    for i in range(len(audio) - window_size - prediction_samples):
        xw = audio[i:i+window_size]
        _, yhat = forward(W1_pred, W2_pred, xw, hidden_gain, use_output_sigmoid)
        predicted_audio.append(yhat)
    predicted_audio = np.array(predicted_audio, dtype=np.float32)      # (T, 32)
    predicted_continuous = predicted_audio[:, 0]                      

    out_pred = f"predicted_audio_ode_{audio_name}.wav"
    wavfile.write(out_pred, sr, np.int16(predicted_continuous * 32767))
    out_orig = f"original_audio_snippet_ode_{audio_name}.wav"
    orig_snippet = audio[:len(predicted_continuous)]
    wavfile.write(out_orig, sr, np.int16(orig_snippet * 32767))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].semilogy(errors_repro, 'b-', lw=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Task 1: Reproduction (ODE Training)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(range(len(errors_pred_full)), errors_pred_full, 'r-', lw=2)
    axes[0, 1].axvline(x=len(errors_pred_stage1)-1, color='k', ls='--', label='Stage 1→2')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Task 2: Prediction (ODE Training)')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)


    axes[1, 0].semilogy(evals, 'go-', lw=2, ms=4)
    axes[1, 0].set_xlabel('Eigenvalue Index'); axes[1, 0].set_ylabel('Eigenvalue')
    axes[1, 0].set_title(f'Covariance Eigenspectrum (spread: {spread:.2e})')
    axes[1, 0].grid(True, alpha=0.3)

    T = min(5000, len(audio))
    time_axis = np.arange(T) / sr
    axes[1, 1].plot(time_axis, audio[:T], lw=0.5)
    axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Audio Waveform')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'audio_network_results_ode_{T}_run{audio_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Final reproduction error: {errors_repro[-1]:.6f}")
    print(f"Final prediction error: {errors_pred_full[-1]:.6f}")
    print(f"Hidden layer nodes used: {H}")
    print("ODE Integration: Euler (dt=1.0)")
    print("Timescale ratio: τ_slow/τ_fast = 100")
