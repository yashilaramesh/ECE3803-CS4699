import nengo
import numpy as np

# x' = cos(x) + 5x^2 - 3
def f(t, x, u):
    return np.cos(x) + 5*x**2 - 3

#Low Pass Filter 
# y' = tanh(x-y)
# x = k/2U_t
# y = k/2U_t
# tau = 2(C_l*U_t)/k(I_bias)
# U_t = 25.8 mV (room temp)
# k = 0.7 (process dependent)
k = 0.7
U = 25.8
C = 100 #load capacitance

def y_prime(k,U):
    x = k/(2*U)
    y = k /(2*U)
    return np.tanh(x-y)

def rk4_step(f, t, x, dt):
    k1 = f(t, x, None)
    k2 = f(t + dt/2, x + dt*k1/2, None)
    k3 = f(t + dt/2, x + dt*k2/2, None)
    k4 = f(t + dt,   x + dt*k3,   None)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6



with nengo.Network() as net:
    dt = 0.001
    state = {"x": np.array(0.5), "t": 0.0}

    def ode_node(t):
        x = state["x"]
        state["x"] = rk4_step(f, t, x, dt)
        state["t"] = t
        return [state["x"]]

    x_node = nengo.Node(ode_node, size_out=1)
    x_probe = nengo.Probe(x_node, synapse=None)

sim = nengo.Simulator(net, dt=0.001)
sim.run(2.0)

import matplotlib.pyplot as plt
plt.plot(sim.trange(), sim.data[x_probe])
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()


