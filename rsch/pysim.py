import numpy as np
import matplotlib.pyplot as plt

# -------- Parameters --------
C_L   = 50e-12       # load capacitance [F]
I_bias= 50e-9        # bias current [A]
V_L   = 50e-3        # scaling/linearity voltage [V]
v0    = 0.0          # initial v_out [V]

# Derived small-signal time constant
tau = C_L * V_L / I_bias
print(f"Small-signal tau ≈ {tau*1e3:.3f} ms")

# -------- Input signals --------
def vin_step(t, Vlow=0.0, Vhigh=0.2, tstep=1e-3):
    return Vlow if t < tstep else Vhigh

def vin_sine(t, A=0.1, f=500):      # 500 Hz sine, A is amplitude
    return A * np.sin(2*np.pi*f*t)

# -------- ODE right-hand sides --------
def f_nonlinear(t, v, vin):
    return (I_bias / C_L) * np.tanh((vin - v) / V_L)

def f_linear(t, v, vin):
    # linearized small-signal model: dv/dt = (1/tau)*(vin - v)
    return (1.0 / tau) * (vin - v)

# -------- RK4 integrator --------
def rk4(deriv, t, v0, vin_fun):
    v = np.empty_like(t)
    v[0] = v0
    for k in range(len(t)-1):
        h = t[k+1]-t[k]
        v_in = vin_fun(t[k])
        k1 = deriv(t[k],     v[k],             v_in)
        k2 = deriv(t[k]+h/2, v[k]+h*k1/2,      vin_fun(t[k]+h/2))
        k3 = deriv(t[k]+h/2, v[k]+h*k2/2,      vin_fun(t[k]+h/2))
        k4 = deriv(t[k]+h,   v[k]+h*k3,        vin_fun(t[k]+h))
        v[k+1] = v[k] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return v

# -------- Simulation helpers --------
def simulate(vin_fun, T=5e-3, dt=1e-6):
    t = np.arange(0, T+dt, dt)
    vnl = rk4(f_nonlinear, t, v0, vin_fun)
    vlin= rk4(f_linear,    t, v0, vin_fun)
    vin = np.vectorize(vin_fun)(t)
    return t, vin, vnl, vlin

# -------- Run: Step response --------
t1, vin1, vnl1, vlin1 = simulate(lambda tt: vin_step(tt, Vhigh=0.3, tstep=0.5e-3),
                                 T=6e-3, dt=2e-6)

plt.figure()
plt.title("FG-TA LPF: Step Response (nonlinear vs linearized)")
plt.plot(t1*1e3, vin1, label="v_in")
plt.plot(t1*1e3, vnl1, label="v_out (nonlinear)")
plt.plot(t1*1e3, vlin1, label="v_out (linearized)", linestyle="--")
plt.xlabel("time [ms]")
plt.ylabel("voltage [V]")
plt.legend()
plt.grid(True)

# -------- Run: Sine response --------
t2, vin2, vnl2, vlin2 = simulate(lambda tt: vin_sine(tt, A=0.08, f=1e3),
                                 T=5e-3, dt=2e-6)

plt.figure()
plt.title("FG-TA LPF: 1 kHz Sine Response")
plt.plot(t2*1e3, vin2, label="v_in")
plt.plot(t2*1e3, vnl2, label="v_out (nonlinear)")
plt.plot(t2*1e3, vlin2, label="v_out (linearized)", linestyle="--")
plt.xlabel("time [ms]")
plt.ylabel("voltage [V]")
plt.legend()
plt.grid(True)

plt.show()
