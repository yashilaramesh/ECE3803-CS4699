import nengo
import numpy as np

def ode_block(
    f,
    n_state,           # dimension of x
    n_input=0,         # dimension of u
    x0=None,           # initial state
    tau=2,          # lowpass for state dynamics (s)
    N_state=800,       # neurons for state representation
    N_ctx=1200,        # neurons for [x; u] context
    neuron_type=None,
    radius_x=1.5,      # expected |x| range
    radius_ctx=2.0,    # expected |[x;u]| range
):
    if neuron_type is None:
        neuron_type = nengo.LIF()

    with nengo.Network(label="SpikingODE") as net:
        if n_input > 0:
            u = nengo.Node(lambda t, x: x, size_in=n_input, size_out=n_input, label="u")
        else:
            u = None

       # State ensemble
        x = nengo.Ensemble(
            n_neurons=N_state,
            dimensions=n_state,
            neuron_type=neuron_type,
            radius=radius_x,
            label="state_x",
        )

        if x0 is not None:
            x_init = nengo.Node(lambda t: x0, size_out=n_state, label="x_init")
            nengo.Connection(x_init, x, synapse=tau)

        # Context ensemble
        ctx_dims = n_state + (n_input if n_input > 0 else 0)
        ctx = nengo.Ensemble(
            n_neurons=N_ctx,
            dimensions=ctx_dims,
            neuron_type=neuron_type,
            radius=radius_ctx,
            label="ctx_[x;u]",
        )

        # Wire x and u into ctx
        nengo.Connection(x, ctx[:n_state], synapse=None)
        if u is not None:
            nengo.Connection(u, ctx[n_state:], synapse=None)
        def dyn_func(ctx_val):
            xv = ctx_val[:n_state]
            if n_input > 0:
                uv = ctx_val[n_state:]
            else:
                uv = np.zeros(0)
            return xv + tau * f(xv, uv)

        nengo.Connection(ctx, x, function=dyn_func, synapse=tau)

        # Probes
        x_probe = nengo.Probe(x, synapse=0.01)

        # Handles
        net.x = x
        net.u = u
        net.x_probe = x_probe

    return net


# LPF (first-order, FG TA form)
# ODE: C * dVout/dt = I_bias * tanh((Vin - Vout)/V_L)
# State: x = [Vout], Input: u = [Vin]

def LPF(
    C=1e-9, I_bias=50e-9, V_L=0.05,
    tau_impl=2,
    N_state=600, N_ctx=900,
    radius_x=1.5, radius_ctx=2.0,
    x0=0.0,
):
    def f(x, u):
        Vout = x[0]
        Vin = u[0] if u.size else 0.0
        dV = (I_bias / C) * np.tanh((Vin - Vout) / V_L)
        return np.array([dV])

    return ode_block(
        f=f, n_state=1, n_input=1, x0=[x0], tau=tau_impl,
        N_state=N_state, N_ctx=N_ctx,
        radius_x=radius_x, radius_ctx=radius_ctx
    )


# C4 Band-Pass (derivative-free state form)
# States: y1 = Vout_a, y2 = V1
#   C_eq * dy1/dt = ((C1+C2+Cw)/C2) * I1 + I2
#   C_eq * dy2/dt = I2 + ((CL+C2)/C2) * I1
# Currents:
#   I1 = I_bias * tanh( -(kappa/(2*Ut)) * V1_a )
#   I2 = Ifb    * tanh( (Vout_a - V1_a - (C1/Ceq)*(CL/C2)*Vin)/V_L )
# where:
#   V1_a   = V1 - (C1/Ceq)*((CL+C2)/C2)*Vin
#   Vout_a = y1 (state)
# Recover: Vout = y1 + (C1/Ceq)*Vin
def BPF(
    C1=2e-9, C2=2e-9, Cw=0.5e-9, CL=2e-9,
    I_bias=50e-9, Ifb=50e-9,
    V_L=0.05, kappa=0.7, Ut=0.0258,
    tau_impl=0.002,
    N_state=900, N_ctx=1400,
    radius_x=1.5, radius_ctx=2.5,
    x0=(0.0, 0.0),
):
    Ceq = (C1 + C2 + Cw) * (CL + C2) - (C2 ** 2)
    a1 = (C1 / Ceq) * ((CL + C2) / C2)  # coefficient for Vin in V1_a
    a2 = (C1 / Ceq)                      # coefficient for Vin in Vout reconstruction
    a3 = (C1 / Ceq) * (CL / C2)          # coefficient in I2 argument

    def f(x, u):
        y1 = x[0]
        y2 = x[1]
        Vin = u[0] if u.size else 0.0

        V1_a = y2 - a1 * Vin
        Vout_a = y1

        I1 = I_bias * np.tanh((-kappa / (2 * Ut)) * V1_a)
        I2 = Ifb    * np.tanh((Vout_a - V1_a - a3 * Vin) / V_L)

        dy1 = ( ((C1 + C2 + Cw) / C2) * I1 + I2 ) / Ceq
        dy2 = ( I2 + ((CL + C2) / C2) * I1 ) / Ceq
        return np.array([dy1, dy2])

    net = ode_block(
        f=f, n_state=2, n_input=1, x0=list(x0), tau=tau_impl,
        N_state=N_state, N_ctx=N_ctx,
        radius_x=radius_x, radius_ctx=radius_ctx
    )

    with net:
        Vrec = nengo.Node(lambda t, inp: [inp[0] + a2 * inp[1]],
                          size_in=2, size_out=1, label="Vout_recon")
        nengo.Connection(net.x[0], Vrec[0], synapse=None)
        nengo.Connection(net.u,    Vrec[1], synapse=None) 
        net.vout_probe = nengo.Probe(Vrec, synapse=0.01)

    return net


# Hopf LPF (two first-order ODEs)
#   tau * dVout/dt + Vout = V
#   tau * dV/dt   + V    = Vin + a*V_L * tanh(kappa*(V - Vout)/(2*U_T))
# State: x = [Vout, V], Input: u = [Vin]

# def Hopf(
#     tau=0.01, a=1.2, V_L=0.05, kappa=0.7, Ut=0.0258,
#     tau_impl=0.002,
#     N_state=800, N_ctx=1200,
#     radius_x=1.5, radius_ctx=2.0,
#     x0=(0.0, 0.0),
# ):
#     def f(x, u):
#         Vout, V = x
#         Vin = u[0] if u.size else 0.0
#         dVout = (-Vout + V) / tau
#         dV    = (-V + (Vin + a * V_L * np.tanh(kappa * (V - Vout) / (2 * Ut)))) / tau
#         return np.array([dVout, dV])

#     return ode_block(
#         f=f, n_state=2, n_input=1, x0=list(x0), tau=tau_impl,
#         N_state=N_state, N_ctx=N_ctx,
#         radius_x=radius_x, radiusx_ctx=radius_ctx
#     )


if __name__ == "__main__":
    DEMO = "LPF"

    if DEMO == "LPF":
        net = LPF()
        with net:
            src = nengo.Node(lambda t: [0.4 * np.sin(2 * np.pi * 5.0 * t)],
                             size_out=1, label="Vin_src")
            nengo.Connection(src, net.u, synapse=None)
            p_in = nengo.Probe(src, synapse=None)

    elif DEMO == "BPF":
        net = BPF()
        with net:
            def vin(t):
                return [0.2 * np.sin(2 * np.pi * 30 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)]
            src = nengo.Node(vin, size_out=1, label="Vin_src")
            nengo.Connection(src, net.u, synapse=None)
            p_in = nengo.Probe(src, synapse=None)

    elif DEMO == "HOPF":
        net = Hopf()
        with net:
            src = nengo.Node(lambda t: [0.3 * np.sin(2 * np.pi * 10 * t)],
                             size_out=1, label="Vin_src")
            nengo.Connection(src, net.u, synapse=None)
            p_in = nengo.Probe(src, synapse=None)

    sim = nengo.Simulator(net, dt=0.001)
    sim.run(2.0)

    import matplotlib.pyplot as plt
    t = sim.trange()

    plt.figure()
    plt.plot(t, sim.data[p_in], label="Vin")

    if DEMO == "LPF":
        plt.plot(t, sim.data[net.x_probe][:, 0], label="LPF Vout")
    elif DEMO == "BPF":
        plt.plot(t, sim.data[net.vout_probe][:, 0], label="BPF Vout (reconstructed)")
    elif DEMO == "HOPF":
        plt.plot(t, sim.data[net.x_probe][:, 0], label="Hopf Vout")
        # plt.plot(t, sim.data[net.x_probe][:, 1], label="Hopf V (internal)")

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.show()
