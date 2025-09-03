import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# constants
N = 16                      # input dimension
f = 1000.0                  # Hz
tau = 2e-3                  # seconds (0.2 ms)
T = 5.0                     # total simulation time (s)
dt_sample = 1e-5                  
thetas = np.random.uniform(low=0, high=2*np.pi, size=N)  # random phases

# random initial condition
w0 = np.random.randn(N) * 0.01
omega = 2*np.pi*f

def x_of_t(t):
    return np.cos(omega*t + thetas)

def wdot(t, w):
    x = x_of_t(t)
    y = float(np.dot(w, x))
    return (y * x - (y**2) * w) / tau

t_eval = np.arange(0.0, T + dt_sample, dt_sample)

sol = spi.solve_ivp(
    fun=wdot,
    t_span=(0.0, T),
    y0=w0,
    method='RK23',
    max_step=dt_sample,
    rtol=1e-6,
    atol=1e-8
)
print(sol)
W = sol.y.T
val = [np.dot(W[n], x_of_t(sol.t[n])) for n in range(len(sol.t))]
skip = max(1, len(sol.t) // 2000)
X = np.cos(omega * sol.t[:, None] + thetas[None, :])



# Plot w(t) and y(t)
plt.figure(figsize=(10,5))
for j in range(N):
    plt.plot(sol.t[::skip], sol.y[j, ::skip], label=f"w{j}",linewidth=1)
plt.xlabel("Time (sec)")
plt.ylabel("w(t)")
plt.title("w(t) graph")
plt.legend(ncol=2, fontsize=8)

plt.show()

# y(t) line
plt.figure(figsize=(10,3))
plt.plot(sol.t[::skip], np.array(val)[::skip])
plt.xlabel("Time (sec)")
plt.ylabel("y(t)")
plt.title("y(t) graph")

plt.show()