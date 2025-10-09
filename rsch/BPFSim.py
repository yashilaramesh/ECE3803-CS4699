import numpy as np
import matplotlib.pyplot as plt

# -------- Parameters --------
C     = 50e-12        # integrator capacitance [F]
V_L   = 50e-3         # FG scaling voltage [V]
Q     = 2.0           # quality factor (shape of the band)

f0    = 3_000.0       # target center frequency [Hz]
w0    = 2*np.pi*f0

I_bias = w0 * C * V_L

print(f"w0 = {w0:.1f} rad/s, f0 = {f0:.1f} Hz,  I_bias ≈ {I_bias*1e9:.2f} nA")

def u_step(t, A=0.2, tstep=1e-3):
    return 0.0 if t < tstep else A

def u_sine(t, A=0.1, f=1_000.0):
    return A * np.sin(2*np.pi*f*t)

#ODE right-hand sides
def fgta_bpf_rhs(t, state, u):
    LP, BP = state
    dLP = (I_bias / C) * np.tanh(BP / V_L)
    dBP = (I_bias / C) * np.tanh((u - LP - (BP / Q)) / V_L)
    return np.array([dLP, dBP])

def linear_bpf_rhs(t, state, u):
    LP, BP = state
    dLP = w0 * BP
    dBP = w0 * (u - LP - (BP / Q))
    return np.array([dLP, dBP])

#rk4
def rk4_vec(rhs, t, x0, ufun):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for k in range(len(t)-1):
        h = t[k+1] - t[k]
        u1 = ufun(t[k])
        k1 = rhs(t[k], x[k], u1)
        k2 = rhs(t[k]+h/2, x[k]+h*k1/2, ufun(t[k]+h/2))
        k3 = rhs(t[k]+h/2, x[k]+h*k2/2, ufun(t[k]+h/2))
        k4 = rhs(t[k]+h,   x[k]+h*k3,   ufun(t[k]+h))
        x[k+1] = x[k] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x

def simulate(ufun, T=6e-3, dt=2e-6, x0=(0.0, 0.0)):
    t = np.arange(0, T+dt, dt)
    x_fg  = rk4_vec(fgta_bpf_rhs,   t, np.array(x0), ufun)
    x_lin = rk4_vec(linear_bpf_rhs, t, np.array(x0), ufun)
    u = np.vectorize(ufun)(t)
    y_fg  = x_fg[:,1]   # BP output
    y_lin = x_lin[:,1]
    return t, u, y_fg, y_lin

t1, u1, y1_fg, y1_lin = simulate(lambda tt: u_step(tt, A=0.3, tstep=0.5e-3),
                                 T=8e-3, dt=2e-6)

plt.figure()
plt.title("Band-Pass Filter (SVF): Step Response")
plt.plot(t1*1e3, u1,     label="u_in")
plt.plot(t1*1e3, y1_fg,  label="y_out (FG-TA nonlinear)")
plt.plot(t1*1e3, y1_lin, label="y_out (linear)", linestyle="--")
plt.xlabel("time [ms]"); plt.ylabel("voltage [V]")
plt.legend(); plt.grid(True)

#sine at f0
t2, u2, y2_fg, y2_lin = simulate(lambda tt: u_sine(tt, A=0.08, f=f0),
                                 T=5e-3, dt=2e-6)

plt.figure()
plt.title(f"Band-Pass: Sine at f0 = {f0:.0f} Hz")
plt.plot(t2*1e3, u2,     label="u_in")
plt.plot(t2*1e3, y2_fg,  label="y_out (FG-TA nonlinear)")
plt.plot(t2*1e3, y2_lin, label="y_out (linear)", linestyle="--")
plt.xlabel("time [ms]"); plt.ylabel("voltage [V]")
plt.legend(); plt.grid(True)

#off-resonance sine
t3, u3, y3_fg, y3_lin = simulate(lambda tt: u_sine(tt, A=0.08, f=200.0),
                                 T=15e-3, dt=2e-6)

plt.figure()
plt.title("Band-Pass: Off-Resonance Sine (200 Hz)")
plt.plot(t3*1e3, u3,     label="u_in")
plt.plot(t3*1e3, y3_fg,  label="y_out (FG-TA nonlinear)")
plt.plot(t3*1e3, y3_lin, label="y_out (linear)", linestyle="--")
plt.xlabel("time [ms]"); plt.ylabel("voltage [V]")
plt.legend(); plt.grid(True)

plt.show()
