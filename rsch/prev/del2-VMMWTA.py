import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
class VMMWTA:
    def __init__(self, n_inputs=4):
        self.n = n_inputs

        self.U_T = 0.0258
        self.I_bias = 1e-9
        self.C_L = 100e-15
        self.kappa_n = 0.7
        self.V_bias = 0.5

        self.V_k_init = np.zeros(n_inputs)
        self.V_d_init = np.zeros(n_inputs)
        self.V_out_init = np.zeros(n_inputs)
        self.I_prog = np.ones(n_inputs) * self.I_bias

    def voltage_input(self, V_k):
        #equation 24
        exp_terms = np.exp(self.kappa_n * V_k/self.U_T)
        V = self.U_T * np.log(np.sum(exp_terms)) - self.V_bias
        return V
    
    def cascode(self, V_d):
        #equation 26
        exp_terms = np.exp(self.kappa_n * V_d / self.U_T)
        V = self.U_T * np.log(np.sum(exp_terms)) - self.V_bias
        return V
    
    def ode_voltage_input(self, state, t, V_d_input):
        #equations 24-25

        V_k = state[:self.n]
        V_out = state[self.n:]

        tau = self.C_L * self.U_T/self.I_bias
        V = self.voltage_input(V_k)

        #common-gate configuration before Eq. 24
        dV_k_dt = np.zeros(self.n)
        for k in range(self.n):
            exp_V_d = np.exp(V_d_input[k] / self.U_T)
            exp_V = np.exp(-self.kappa_n * V / self.U_T)
            exp_V_k = np.exp(V_k[k] / self.U_T)
            
            dV_k_dt[k] = (self.I_bias / self.C_L) * (
                exp_V_d - exp_V * (1 - exp_V_k)
            )

        dv_out_dt = np.zeros(self.n)
        exp_V_k = np.exp(V_k)
        sum_exp_V = np.sum(exp_V_k)

        for k in range(self.n):
            V_dd = 3.3
            term1 = self.I_prog[k] * np.exp((V_dd - V_out[k])/self.U_T)
            term2 = self.I_bias * exp_V_k[k]/sum_exp_V

            dv_out_dt[k] = (1/self.C_L) * (term1-term2)
        return np.concatenate([dV_k_dt, dv_out_dt])
    def ode_cascode(self,state,t, V_d_input):
        #Equation 26
        V_out = state
        V = self.cascode(V_d_input)
        dV_out_dt = np.zeros(self.n)
        exp_V_d = np.exp(V_d_input)
        sum_V_d = np.sum(exp_V_d)

        V_dd = 3.3
        for k in range(self.n):
            term1 = self.I_prog[k] * np.exp((V_dd-V_out[k])/self.U_T)
            term2 = self.I_bias * exp_V_d[k]/sum_V_d
            dV_out_dt[k] =(1/self.C_L) * (term1-term2)

        return dV_out_dt
    
    def simulate(self, V_d_input_func, t_span, dt=1e-6):
        t = np.arange(t_span[0], t_span[1], dt)
        state0 = self.V_out_init
        V_d_input_array = np.array([V_d_input_func(ti) for ti in t])
        results = []
        state = state0
        for i in range(len(t)):
            if i == 0:
                results.append(state)
            else:
                sol = odeint(self.ode_cascode,state,[t[i-i],t[i]], args=(V_d_input_array[i],))
                state = sol[-1]
                results.append(state)

        V_out = np.array(results)
        return t, {'V_out': V_out, 'V_d_input':V_d_input_array}

if __name__ == "__main__":
    sim = VMMWTA(n_inputs=4)
    #currents out of VMM plot and inputs should be double the amplitudes
    #plot of V (middle node) over t
    def V_d_input_sine(t):
        base = 0.15
        amplitude = 0.10
        freq = 250
        
        phase_0 = 0
        phase_1 = np.pi / 2
        phase_2 = np.pi
        phase_3 = 3 * np.pi / 2
        
        return np.array([
            base + amplitude * np.sin(2 * np.pi * freq * t + phase_0),
            base + amplitude * np.sin(2 * np.pi * freq * t + phase_1),
            base + amplitude * np.sin(2 * np.pi * freq * t + phase_2),
            base + amplitude * np.sin(2 * np.pi * freq * t + phase_3)
        ])
    

    t, results = sim.simulate(V_d_input_sine, t_span=(0, 12e-3), dt=1e-6)
    

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].set_title('V_d')
    for i in range(sim.n):
        axes[0,0].plot(t * 1e3, results['V_d_input'][:, i], label=f'V_d[{i}]')
    axes[0,0].set_ylabel('Voltage (V)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    axes[0,1].set_title('V_out')
    for i in range(sim.n):
        axes[0,1].plot(t * 1e3, results['V_out'][:, i], label=f'V_out[{i}]', linewidth=2)
    axes[0,1].set_xlabel('Time (ms)')
    axes[0,1].set_ylabel('Voltage (V)')
    axes[0,1].legend()
    axes[0,1].grid(True)

    
    plt.tight_layout()
    plt.show()
    
    print("Simulation complete!")
    print(f"Final output voltages: {results['V_out'][-1]}")
    print(f"Winner index: {np.argmax(results['V_d_input'][-1])}")