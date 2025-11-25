
"""
demo_two_spirals.py
Quick demo: train TwoLayerNet on a tiny two-spirals classification (labels -1/+1).
Runs fast with small data for sanity-checking your pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
from backprop_ode import TwoLayerNet, mse_loss

def two_spirals(n_points=200, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = n_points // 2
    theta = np.sqrt(rng.random(n)) * 4 * np.pi
    r = 2 * theta + np.pi
    x1 = np.vstack([
        r*np.cos(theta) + noise*rng.standard_normal(n),
        r*np.sin(theta) + noise*rng.standard_normal(n)
    ]).T
    x2 = np.vstack([
        -r*np.cos(theta) + noise*rng.standard_normal(n),
        -r*np.sin(theta) + noise*rng.standard_normal(n)
    ]).T
    X = np.vstack([x1, x2])
    y = np.hstack([np.ones(n), -np.ones(n)])
    return X, y

def main():
    X, y = two_spirals(n_points=400, noise=0.1, seed=1)
    # map labels {-1,+1} -> {0,1} for sigmoid output if desired, but we'll regress to {-1,+1}
    X = X.T  # (2, N)
    Y = y.reshape(1, -1)  # (1, N)

    net = TwoLayerNet(
        n_in=2, n_hidden=16, n_out=1,
        hidden_activation="tanh",
        output_activation="linear",   # we'll train to targets -1/+1
        gain_hidden=2.0,
        weight_scale=0.5,
        seed=42
    )

    hist = net.fit(
        X, Y,
        epochs=2000,
        batch_size=64,
        eta=0.02,
        dt_w=1.0,
        verbose_every=0
    )
    print(f"final MSE: {hist['loss'][-1]:.4e}")

    # simple accuracy check by sign
    Yh, _ = net.forward(X)
    y_pred = np.sign(Yh).ravel()
    acc = np.mean(y_pred == Y.ravel())
    print(f"train accuracy (sign): {acc*100:.1f}%")

    # plot loss
    plt.figure()
    plt.plot(hist['loss'])
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Two-Spirals training loss")
    plt.tight_layout()
    out = "/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/figures/two_spirals_loss.png"
    plt.savefig(out)
    print(f"Saved loss curve to {out}")

if __name__ == "__main__":
    main()
