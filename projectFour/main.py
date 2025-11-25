import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

AUDIO_PATH      = "/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/ironic.wav"
TARGET_SR       = 16000
TAPS            = 128
PRED_AHEAD      = 8
H               = 2
EPOCHS          = 63
EXAMPLES_PER_EP = 720_000
BATCH_SIZE      = 2048
STEPS_PER_EPOCH = EXAMPLES_PER_EP // BATCH_SIZE
ETA             = 0.01
TAU_SLOW        = 100.0
STEP            = ETA / TAU_SLOW
GAIN            = 2.0

sr, audio = wavfile.read(AUDIO_PATH)

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

if sr != TARGET_SR:
    num = int(len(audio) * TARGET_SR / sr)
    audio = signal.resample(audio, num).astype(np.float32)
    sr = TARGET_SR

audio /= (np.max(np.abs(audio)) + 1e-8)

N = len(audio)
K = N - TAPS - PRED_AHEAD + 1


print(f"Audio length: {N} samples ({N/sr:.2f}s) @ {sr} Hz. Usable training positions: {K}")

rng  = np.random.default_rng(0)
W1   = rng.normal(0.0, 0.5/np.sqrt(TAPS), size=(H, TAPS)).astype(np.float32)
b1   = np.zeros((H, 1), dtype=np.float32)
W2   = rng.normal(0.0, 0.5/np.sqrt(H),    size=(1, H)).astype(np.float32)
b2   = np.zeros((1, 1), dtype=np.float32)

# tanh and derivative (sech^2 = 1 - tanh^2)
def tanh(z):
    return np.tanh(GAIN*z)

def sech2_from_tanh(a):
    return GAIN *(1.0 - a*a)

def make_delay_batch(x, start, B):
    t = np.arange(start, start+B, dtype=np.int64) + (TAPS - 1)
    Xb = np.empty((TAPS, B), dtype=np.float32)
    for i in range(TAPS):
        Xb[i, :] = x[t - i]
    yb = x[t + PRED_AHEAD][None, :].astype(np.float32)
    return Xb, yb


rng_ep = np.random.default_rng(1)
loss_history = []

for ep in range(EPOCHS):
    epoch_losses = []
    for _ in range(STEPS_PER_EPOCH):
        # np.random.permutation(len(X_train))
        start = int(rng_ep.integers(0, K - BATCH_SIZE))
        Xb, yb = make_delay_batch(audio, start, BATCH_SIZE)  

        Z1  = (W1 @ Xb)
        H1  = tanh(Z1)          
        Z2  = (W2 @ H1)
        Yh  = Z2

        E   = (Yh - yb)
        mse = float(np.mean(E*E))
        epoch_losses.append(mse)

        # dL/dYh = 2/B * (Yh - yb)
        dL_dY = (2.0 / BATCH_SIZE) * E

        dW2 = dL_dY @ H1.T
        db2 = np.mean(dL_dY, axis=1, keepdims=True)

        dH  = W2.T @ dL_dY
        dZ1 = dH * sech2_from_tanh(H1)
        dW1 = dZ1 @ Xb.T
        db1 = np.mean(dZ1, axis=1, keepdims=True)

        W2 -= STEP * dW2
        b2 -= STEP * db2
        W1 -= STEP * dW1
        b1 -= STEP * db1

    avg_mse = float(np.mean(epoch_losses))
    loss_history.append(avg_mse)
    print(f"epoch {ep+1:02d}/{EPOCHS}  MSE={avg_mse:.6e}")

Tstream = min(1_000_000, K)
pred_stream = np.empty((Tstream,), dtype=np.float32)

for i in range(Tstream):
    t = i + (TAPS - 1)
    xw = audio[t-(TAPS-1): t+1]
    xw = xw[::-1]
    Xc = xw.reshape(TAPS, 1)

    Z1 = (W1 @ Xc) + b1
    H1 = tanh(Z1)
    Z2 = (W2 @ H1) + b2
    pred_stream[i] = float(Z2[0, 0])

ref = audio[TAPS-1 + PRED_AHEAD : TAPS-1 + PRED_AHEAD + Tstream]

mse_align = float(np.mean((pred_stream - ref)**2))
corr_align = float(np.corrcoef(pred_stream, ref)[0, 1])
print(f"Aligned eval  (window={TAPS}, +{PRED_AHEAD} ahead):  MSE={mse_align:.6e}  Corr={corr_align:.4f}")


def to_int16(x):
    x_clipped = np.clip(x, -1.0, 1.0)
    return np.int16(x_clipped * 32767)

wavfile.write(f"/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/projResults/predicted_{PRED_AHEAD}_ep{EPOCHS}_stream.wav",      sr, to_int16(pred_stream))
wavfile.write(f"/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/projResults/reference_aligned_{PRED_AHEAD}_ep{EPOCHS}.wav",     sr, to_int16(ref))

plt.figure(figsize=(12,5))
plt.semilogy(loss_history, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE (log)")
plt.title("Training Loss (MSE per Epoch)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/projResults/loss_curve_{PRED_AHEAD}_ep{EPOCHS}.png", dpi=200)
print("Saved: loss_curve.png")

seconds_to_plot = 2.0
S = int(min(Tstream, seconds_to_plot * sr))
t_axis = np.arange(S) / sr

plt.figure(figsize=(12,5))
plt.plot(t_axis, audio[:S], label="Reference", linewidth=1)
plt.plot(t_axis, pred_stream[:S], label="Prediction (+8)", linewidth=1, alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Prediction vs Reference (first {seconds_to_plot:.1f}s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/projResults/prediction_overlay_{PRED_AHEAD}_ep{EPOCHS}.png", dpi=200)


MAX_WINDOWS = min(200_000, K)

print(f"\nConstructing covariance matrix on {MAX_WINDOWS} windows of size {TAPS}...")

X_cov = np.empty((MAX_WINDOWS, TAPS), dtype=np.float32)

for i in range(MAX_WINDOWS):
    X_cov[i, :] = audio[i : i + TAPS]

X_mean = np.mean(X_cov, axis=0, keepdims=True)
X_centered = X_cov - X_mean

cov_matrix = np.cov(X_centered, rowvar=False)

eigvals = np.linalg.eigvalsh(cov_matrix)
eigvals_sorted = np.sort(eigvals)[::-1]

lam_max = eigvals_sorted[0]
lam_min = eigvals_sorted[-1]
eig_spread = lam_max / (lam_min + 1e-12)

print("\n=== Covariance / Eigenvalue Analysis ===")
print(f"Top 5 eigenvalues:    {eigvals_sorted[:5]}")
print(f"Bottom 5 eigenvalues: {eigvals_sorted[-5:]}")
print(f"Eigenvalue spread λ_max/λ_min ≈ {eig_spread:.2e}")

np.savetxt(
    f"/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/projResults/"
    f"cov_eigs_TAPS{TAPS}_PRED{PRED_AHEAD}.txt",
    eigvals_sorted,
    header=f"Eigenvalues (descending) for covariance, TAPS={TAPS}, PRED_AHEAD={PRED_AHEAD}\n"
)


plt.figure(figsize=(6, 4))
plt.semilogy(eigvals_sorted, marker='o')
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue (log scale)")
plt.title(f"Covariance eigenvalue spectrum")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/projResults/"
    f"cov_eigs_TAPS{TAPS}_PRED{PRED_AHEAD}.png",
    dpi=200,
)