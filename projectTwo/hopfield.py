
import numpy as np

# Convert between {0,1} and {-1,1} representations
def to_bipolar(y01: np.ndarray) -> np.ndarray:
    """Map {0,1}^n -> {-1,1}^n"""
    return 2.0 * y01 - 1.0

def from_bipolar(s: np.ndarray) -> np.ndarray:
    """Map {-1,1}^n -> {0,1}^n"""
    return (s + 1.0) / 2.0

def hebbian_weights_from_patterns(patterns_01: np.ndarray, zero_diagonal: bool = True, normalize: bool = True) -> np.ndarray:
    """
    Hebbian learning for Hopfield weights.
    patterns_01: shape (k, n) with entries in {0,1}.
    Returns symmetric W with optional zero diagonal.
    """
    patterns = to_bipolar(patterns_01)
    k, n = patterns.shape
    W = np.zeros((n, n), dtype=float)
    for p in patterns:
        W += np.outer(p, p)
    if normalize and n > 0:
        W /= n
    if zero_diagonal:
        np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    return W


def rk4_step(f, t, x, dt):
    k1 = f(t, x)
    k2 = f(t + 0.5*dt, x + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, x + 0.5*dt*k2)
    k4 = f(t + dt, x + dt*k3)
    return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# du/dt = (-u + W*s + b) / tau,  s = tanh(beta * u)
# E = -0.5 * s^T W s - b^T s

class ContinuousHopfield:
    def __init__(self, W: np.ndarray, tau: float = 1.0, beta: float = 4.0, bias: np.ndarray | None = None, seed: int | None = None):
        """
        W: symmetric weight matrix (n x n)
        tau: time constant
        beta: slope of tanh (inverse "temperature")
        bias: optional bias vector b (n,)
        """
        self.W = 0.5 * (W + W.T)
        self.n = W.shape[0]
        self.tau = float(tau)
        self.beta = float(beta)
        self.b = np.zeros(self.n) if bias is None else bias.astype(float)
        self.rng = np.random.default_rng(seed)

        self.u = np.zeros(self.n, dtype=float)
        self.s = np.tanh(self.beta * self.u)

    def set_state_from_pattern01(self, y01: np.ndarray, noise_std: float = 0.0):
        """Initialize u near the bipolar version of a 0/1 pattern."""
        s_target = to_bipolar(y01.astype(float))
        eps = 1e-5
        s_clip = np.clip(s_target, -1+eps, 1-eps)
        u0 = np.arctanh(s_clip) / self.beta
        if noise_std > 0:
            u0 = u0 + self.rng.normal(0.0, noise_std, size=self.n)
        self.u = u0
        self.s = np.tanh(self.beta * self.u)

    def set_state_random(self, scale: float = 0.5):
        """Random u initialization."""
        self.u = self.rng.normal(0.0, scale, size=self.n)
        self.s = np.tanh(self.beta * self.u)

    def _f(self, t, u):
        s = np.tanh(self.beta * u)
        return (-u + self.W @ s + self.b) / self.tau

    def step(self, t, dt):
        self.u = rk4_step(self._f, t, self.u, dt)
        self.s = np.tanh(self.beta * self.u)

    def run(self, T: float, dt: float, record: bool = True):
        steps = int(np.ceil(T / dt))
        ts = None
        traj_u = None
        traj_s = None
        energies = None
        if record:
            ts = np.zeros(steps+1)
            traj_u = np.zeros((steps+1, self.n))
            traj_s = np.zeros((steps+1, self.n))
            energies = np.zeros(steps+1)
            traj_u[0] = self.u.copy()
            traj_s[0] = self.s.copy()
            energies[0] = self.energy_proxy(self.s)
        t = 0.0
        for k in range(1, steps+1):
            self.step(t, dt)
            t += dt
            if record:
                ts[k] = t
                traj_u[k] = self.u.copy()
                traj_s[k] = self.s.copy()
                energies[k] = self.energy_proxy(self.s)
        return ts, traj_u, traj_s, energies

    def energy_proxy(self, s: np.ndarray | None = None) -> float:
        """Discrete Hopfield-style energy proxy (for monitoring)."""
        if s is None:
            s = self.s
        return -0.5 * float(s @ (self.W @ s)) - float(self.b @ s)

    def hard_readout(self) -> np.ndarray:
        """Return discrete {-1,1} state from current s."""
        return np.sign(self.s + 1e-12)  # avoid zeros


def maxcut(A: np.ndarray, tau: float = 1.0, beta: float = 4.0, seed: int | None = None) -> ContinuousHopfield:
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    W = -A.copy()
    return ContinuousHopfield(W=W, tau=tau, beta=beta, bias=None, seed=seed)

def cut_value(A: np.ndarray, s_sign: np.ndarray) -> float:
    """Compute cut value for spins s in {-1,1}^n and adjacency A."""
    # Cut = sum_{i<j} A_ij * [s_i != s_j] = 0.5 * sum_{i<j} A_ij * (1 - s_i s_j)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    S = np.outer(s_sign, s_sign)
    return float(0.25 * np.sum(A * (1.0 - S)))
