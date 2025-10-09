import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
C      = 50e-12      # integrating capacitance per output [F]
V_L    = 50e-3       # TA scaling voltage [V]
I_bias = 20e-9       # TA bias current [A]
I_s    = I_bias      # slope scale inside tanh
k_L    = 0.6*I_bias  # leak strength
k_I    = 1.2*I_bias  # lateral inhibition strength

# VMM dimensions
N_in, N_out = 4, 5

rng = np.random.default_rng(0)
W = rng.normal(0.0, 1.0, size=(N_out, N_in))
x = np.array([0.2, -0.1, 0.35, 0.05])

# ---------------- VMM stage ----------------
d = W @ x                      
I_in = I_bias * np.tanh(d / V_L) 

print("drives d:", np.round(d, 3))
print("argmax(Wx) =", int(np.argmax(d)))

# ---------------- WTA ODE ----------------
def wta_rhs(t, y):
    S = np.sum(y)
    others = S - y
    u = I_in - k_L * y - k_I * others
    dy = (I_bias / C) * np.tanh(u / I_s)
    return dy

# ---------------- RK4 integrator ----------------
def rk4_vec(rhs, t, y0):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for k in range(len(t)-1):
        h = t[k+1] - t[k]
        k1 = rhs(t[k],       y[k])
        k2 = rhs(t[k]+h/2.0, y[k] + h*k1/2.0)
        k3 = rhs(t[k]+h/2.0, y[k] + h*k2/2.0)
        k4 = rhs(t[k]+h,     y[k] + h*k3)
        y[k+1] = y[k] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        y[k+1] = np.maximum(y[k+1], 0.0)
    return y

# ---------------- Simulate ----------------
T, dt = 6e-3, 2e-6
t = np.arange(0.0, T+dt, dt)
y0 = np.zeros(N_out)
Y = rk4_vec(wta_rhs, t, y0)

winners = np.argmax(Y, axis=1)
final_winner = int(winners[-1])

print("final winner =", final_winner)

# ---------------- Plots ----------------
plt.figure()
for i in range(N_out):
    plt.plot(t*1e3, Y[:, i], label=f"y[{i}]")
plt.xlabel("time [ms]"); plt.ylabel("activity (arb.)")
plt.title("VMM + WTA dynamics")
plt.grid(True); plt.legend(ncol=2)

plt.figure()
plt.bar(np.arange(N_out)-0.2, d, width=0.4, label="drive d = W x")
plt.bar(np.arange(N_out)+0.2, Y[-1], width=0.4, label="final y (WTA)")
plt.xticks(np.arange(N_out))
plt.title(f"Argmax check: argmax(Wx)={int(np.argmax(d))}, winner={final_winner}")
plt.legend()
plt.show()
