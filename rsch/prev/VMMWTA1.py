import numpy as np
import matplotlib.pyplot as plt

UT = 0.0258
def exp_clip(x, lim=40.0):
    return np.exp(np.clip(x, -lim, lim))

def softmax_kappa(v, kappa):
    x = np.clip(kappa * v, -40.0, 40.0)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

N_in, N_out = 4, 5
m = N_in
n = N_out

rng = np.random.default_rng(0)

W = rng.normal(0.0, 1.0, size=(n, m))

x = np.array([0.20, -0.10, 0.35, 0.05])
Vg0 = 0.0
dVg = x

# 𝜏 * dVd/dt + UT * exp((Vd - Vd0)/UT) = UT * Σ_k W * exp(-κ_eff * ΔVg/UT)
kappa_eff   = 0.30
Vd0         = 0.0
Cd          = 50e-15
Ibias_vmm   = 50e-9
tau_vmm     = m * Cd * UT / Ibias_vmm

# CL * dVout/dt = Iprog * exp(-(Vdd - Vout)/UT) - Ibias_wta * softmax(κ_soft * Vd)
CL          = 100e-15
Iprog       = 120e-9 * np.ones(n)
Ibias_wta   = 60e-9
kappa_soft  = 3.0
Vdd         = 1.8

use_sigmoid_gain = False
VA, VL_sig, Vref1 = 0.8, 0.10, 0.0 

tau_out = CL / Ibias_wta
T = 6e-3
dt = min(tau_vmm, tau_out) / 30.0
t  = np.arange(0.0, T + dt, dt)

Vd   = np.zeros(n)
Vout = np.zeros(n)

Vd_hist   = np.zeros((len(t), n))
Vout_hist = np.zeros((len(t), n))
Vd_hist[0] = Vd
Vout_hist[0] = Vout

def vmm_rhs(Vd_vec):
    drive = UT * (W @ exp_clip(-kappa_eff * dVg / UT))
    leak  = UT * exp_clip((Vd_vec - Vd0) / UT)
    return (drive - leak) / tau_vmm

def wta_rhs(Vout_vec, Vd_vec):
    Vcomp = VA * np.tanh((Vd_vec - Vref1) / VL_sig) if use_sigmoid_gain else Vd_vec
    sm = softmax_kappa(Vcomp, kappa_soft)
    src  = Iprog * exp_clip(-(Vdd - Vout_vec) / UT)
    sink = Ibias_wta * sm
    return (src - sink) / CL

def coupled_rhs(state):
    Vd_vec   = state[:n]
    Vout_vec = state[n:]
    dVd_dt   = vmm_rhs(Vd_vec)
    dVo_dt   = wta_rhs(Vout_vec, Vd_vec)
    return np.concatenate([dVd_dt, dVo_dt], axis=0)

state = np.concatenate([Vd, Vout], axis=0)
for k in range(len(t)-1):
    h  = t[k+1] - t[k]
    k1 = coupled_rhs(state)
    k2 = coupled_rhs(state + 0.5*h*k1)
    k3 = coupled_rhs(state + 0.5*h*k2)
    k4 = coupled_rhs(state + h*k3)
    state = state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    Vd   = state[:n]
    Vout = state[n:]
    Vd_hist[k+1]   = Vd
    Vout_hist[k+1] = Vout

def winner_from_Vout(Vout_vec): return int(np.argmax(Vout_vec))
def winner_from_Vd(Vd_vec):    return int(np.argmax(Vd_vec))

print(f"tau_vmm={tau_vmm:.2e}s, tau_out={tau_out:.2e}s, dt={dt:.2e}s")
print("Final Vd:",   np.round(Vd_hist[-1],   4))
print("Final Vout:", np.round(Vout_hist[-1], 4))
print("Softmax(Vd) final:", np.round(softmax_kappa(Vd_hist[-1], kappa_soft), 4))
print("Winner (Vd):",   winner_from_Vd(Vd_hist[-1]))
print("Winner (Vout):", winner_from_Vout(Vout_hist[-1]))

plt.figure()
for i in range(n):
    plt.plot(t*1e3, Vd_hist[:, i], label=f"Vd[{i}]")
plt.xlabel("time [ms]"); plt.ylabel("Vd (V)")
plt.title("VMM drain node dynamics Vd(t)")
plt.grid(True); plt.legend(ncol=2)

plt.figure()
for i in range(n):
    plt.plot(t*1e3, Vout_hist[:, i], label=f"Vout[{i}]")
plt.xlabel("time [ms]"); plt.ylabel("Vout (V)")
plt.title("WTA outputs Vout(t)")
plt.grid(True); plt.legend(ncol=2)

plt.show()
