
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hopfield import (
    hebbian_weights_from_patterns,
    ContinuousHopfield,
    to_bipolar, from_bipolar,
    hopfield_for_maxcut, cut_value
)

outdir = Path("/Users/yashila/Documents/GitHub/NengoBrain/projectTwo/outputs")
outdir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helper plotting functions
# ----------------------------
def plot_pattern(y01, title, savepath):
    fig = plt.figure()
    plt.stem(np.arange(len(y01)), y01, use_line_collection=True)
    plt.ylim([-0.2, 1.2])
    plt.xlabel("Index")
    plt.ylabel("Bit (0/1)")
    plt.title(title)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

def plot_matrix(M, title, savepath):
    fig = plt.figure()
    plt.imshow(M, aspect="equal")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("j")
    plt.ylabel("i")
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

def plot_trajectory(ts, energies, title, savepath):
    fig = plt.figure()
    plt.plot(ts, energies)
    plt.xlabel("t")
    plt.ylabel("Energy (proxy)")
    plt.title(title)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

def plot_state(s, title, savepath, y01=False):
    fig = plt.figure()
    y = from_bipolar(s) if y01 else s
    plt.stem(np.arange(len(y)), y, use_line_collection=True)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(title)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Case 1: Store 3 patterns, n >= 21
# ----------------------------
n = 25
rng = np.random.default_rng(42)

# Create three distinct binary patterns (0/1)
# Here we choose structured patterns for clarity.
p1 = np.zeros(n); p1[:n//2] = 1  # first half ones
p2 = np.zeros(n); p2[::2] = 1    # even indices ones
p3 = (rng.random(n) > 0.5).astype(float)  # random bits

patterns = np.vstack([p1, p2, p3])  # shape (3, n)

# Save pattern plots
plot_pattern(p1, "Pattern 1 (0/1)", outdir / "case1_pattern1.png")
plot_pattern(p2, "Pattern 2 (0/1)", outdir / "case1_pattern2.png")
plot_pattern(p3, "Pattern 3 (0/1)", outdir / "case1_pattern3.png")

# Hebbian weights (using provided formula via bipolar mapping)
W = hebbian_weights_from_patterns(patterns, zero_diagonal=True, normalize=True)
plot_matrix(W, "Weights W (Hebbian, zero diag, normalized)", outdir / "case1_W.png")

# Build network
tau = 1.0
beta = 4.0
net = ContinuousHopfield(W=W, tau=tau, beta=beta, seed=7)

# For each stored pattern, start from a noisy init and show convergence
T = 5.0
dt = 0.01
noise_std = 0.6

for idx, target in enumerate(patterns, start=1):
    net.set_state_from_pattern01(target, noise_std=noise_std)
    ts, _, traj_s, energies = net.run(T=T, dt=dt, record=True)
    # Save energy trajectory
    plot_trajectory(ts, energies, f"Case 1: Energy (Pattern {idx})", outdir / f"case1_energy_p{idx}.png")
    # Save initial vs final states
    s_init = traj_s[0]
    s_final = traj_s[-1]
    plot_state(s_init, f"Case 1: Initial s (Pattern {idx})", outdir / f"case1_init_s_p{idx}.png")
    plot_state(s_final, f"Case 1: Final s (Pattern {idx})", outdir / f"case1_final_s_p{idx}.png")
    # Also save final hard readout in 0/1
    y_final = (np.sign(s_final) + 1.0) / 2.0
    plot_state(np.sign(s_final), f"Case 1: Final sign(s) (Pattern {idx})", outdir / f"case1_final_signs_p{idx}.png")
    plot_state(y_final, f"Case 1: Final y (0/1) (Pattern {idx})", outdir / f"case1_final_y01_p{idx}.png")

# ----------------------------
# Case 2: Max-Cut on small graph (n = 8), three inits
# ----------------------------
m = 8
# Define a small weighted undirected graph
A = np.zeros((m, m), dtype=float)
edges = [
    (0,1,1.0), (0,2,0.8), (1,2,0.5), (1,3,1.2), (2,4,1.1),
    (3,4,0.7), (3,5,1.0), (4,5,0.9), (4,6,0.6), (5,7,1.3),
    (6,7,0.4), (2,6,0.5), (1,7,0.6)
]
for i,j,w in edges:
    A[i,j] = w; A[j,i] = w

# Build Hopfield for Max-Cut: W = -A
maxcut_net = hopfield_for_maxcut(A, tau=1.0, beta=4.0, seed=123)

# Plot adjacency and implied W
fig = plt.figure()
plt.imshow(A, aspect="equal")
plt.colorbar()
plt.title("Case 2: Graph adjacency A")
plt.xlabel("j"); plt.ylabel("i")
fig.savefig(outdir / "case2_adjA.png", bbox_inches="tight")
plt.close(fig)

fig = plt.figure()
plt.imshow(-A, aspect="equal")
plt.colorbar()
plt.title("Case 2: Hopfield W = -A")
plt.xlabel("j"); plt.ylabel("i")
fig.savefig(outdir / "case2_W_minusA.png", bbox_inches="tight")
plt.close(fig)

# Run from three different initial conditions
T2 = 6.0
dt2 = 0.01
inits = [
    ("rand_small", 0.2),
    ("rand_med", 0.8),
    ("rand_big", 1.5),
]

results = []
for name, scale in inits:
    maxcut_net.set_state_random(scale=scale)
    ts, _, traj_s, energies = maxcut_net.run(T=T2, dt=dt2, record=True)
    s_final_sign = np.sign(traj_s[-1] + 1e-12)
    value = cut_value(A, s_final_sign)
    results.append((name, s_final_sign, value, ts, energies, traj_s))

    # Save plots
    plot_trajectory(ts, energies, f"Case 2: Energy ({name})", outdir / f"case2_energy_{name}.png")
    plot_state(traj_s[0], f"Case 2: Initial s ({name})", outdir / f"case2_init_s_{name}.png")
    plot_state(traj_s[-1], f"Case 2: Final s ({name})", outdir / f"case2_final_s_{name}.png")

# Save a small text summary
with open(outdir / "case2_summary.txt", "w") as f:
    for name, s_sign, val, _, _, _ in results:
        f.write(f"Init: {name}\n")
        f.write(f"Final spins: {s_sign.astype(int)}\n")
        f.write(f"Cut value: {val:.3f}\n\n")

print("Done. Outputs written to", outdir)
