import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class VMMWTA:
    def __init__(self, n_inputs=4, n_outputs=4,weight_matrix=None):
        self.n = n_inputs
        self.n_outputs = n_outputs
        self.W = weight_matrix
        self.U_T = 0.0258  # Thermal voltage (V) at room temperature
        self.I_bias =  20e-9  # Bias current (A)
        self.C_L = 100e-15  # Load capacitance (F)
        self.kappa_n = 0.7  # Capacitive coupling coefficient
        self.V_bias = 2.0  # bias voltage
        self.V_dd = 3.0  # supply voltage
        
        # For voltage-input case (Eq. 24)
        self.V_k_init = np.zeros(n_inputs)
        
        # Output voltages
        self.V_out_init = np.zeros(n_inputs)
        self.I_prog = np.ones(n_inputs) * self.I_bias
        
    def extended_diff_pair_voltage_input(self, V_k):
        exp_terms = np.exp(self.kappa_n * V_k / self.U_T)
        V = self.U_T * np.log(np.sum(exp_terms)) - self.V_bias
        return V
    
    def extended_diff_pair_cascode(self, V_d):
        exp_terms = np.exp(self.kappa_n * V_d / self.U_T)
        V = self.U_T * np.log(np.sum(exp_terms)) - self.V_bias
        return V
    
    def compute_vmm_currents(self, V_d_input):
        I_out = np.zeros(self.n_outputs)
        
        for l in range(self.n_outputs):
            for k in range(self.n):
                I_out[l] += self.W[l, k] * np.exp(V_d_input[k] / self.U_T)
        
        I_out = self.I_bias * I_out
        
        return I_out
    def ode_system_voltage_input(self, state, t, V_d_input):
        V_k = state[:self.n]
        V_out = state[self.n:]
        
        #tau = self.C_L * self.U_T / self.I_bias
        
        #equation 24
        V = self.extended_diff_pair_voltage_input(V_k)
        
        # Derivatives for V_k (before Eq. 24)
        dV_k_dt = np.zeros(self.n)
        for k in range(self.n):
            exp_V_d = np.exp(V_d_input[k] / self.U_T)
            exp_V = np.exp(-self.kappa_n * V / self.U_T)
            exp_V_k = np.exp(V_k[k] / self.U_T)
            
            dV_k_dt[k] = (self.I_bias / self.C_L) * (
                exp_V_d - exp_V * (1 - exp_V_k)
            )
        
        # Derivatives for V_out_k (Eq. 25)
        dV_out_dt = np.zeros(self.n)
        exp_V_k = np.exp(V_k)
        sum_exp_V = np.sum(exp_V_k)
        
        for k in range(self.n):
            term1 = self.I_prog[k] * np.exp((self.V_dd - V_out[k]) / self.U_T)
            term2 = self.I_bias * exp_V_k[k] / sum_exp_V
            
            dV_out_dt[k] = (1 / self.C_L) * (term1 - term2)
        
        return np.concatenate([dV_k_dt, dV_out_dt])
    
    def ode_system_cascode(self, state, t, V_d_input):
        V_out = state
        
        # Equation 26
        #V = self.extended_diff_pair_cascode(V_d_input)
        
        # Derivatives for V_out_k (Eq. 25)
        dV_out_dt = np.zeros(self.n)
        exp_V_d = np.exp(V_d_input / self.U_T)
        sum_exp_V_d = np.sum(exp_V_d)
        
        for k in range(self.n):
            term1 = self.I_prog[k] * np.exp((self.V_dd - V_out[k]) / self.U_T)
            term2 = self.I_bias * exp_V_d[k] / sum_exp_V_d
            
            dV_out_dt[k] = (1 / self.C_L) * (term1 - term2)
        
        return dV_out_dt
    
    def simulate(self, V_d_input_func, t_span, dt=1e-6):
        t = np.arange(t_span[0], t_span[1], dt)
    
        state0 = self.V_out_init
        
        V_d_input_array = np.array([V_d_input_func(ti) for ti in t])
        
        results = []
        state = state0
        I_out_array = []
        V_middle_array = []
        
        for i in range(len(t)):
            I_out = self.compute_vmm_currents(V_d_input_array[i])
            V_middle = self.extended_diff_pair_cascode(V_d_input_array[i])
            I_out_array.append(I_out)
            V_middle_array.append(V_middle)
            
            if i == 0:
                results.append(state)
            else:
                sol = odeint(self.ode_system_cascode, state, 
                            [t[i-1], t[i]], args=(V_d_input_array[i],))
                state = sol[-1]
                results.append(state)
        
        V_out = np.array(results)
        I_out_array = np.array(I_out_array)
        V_middle_array = np.array(V_middle_array)
        
        return t, {
            'V_out': V_out, 
            'V_d_input': V_d_input_array,
            'I_out': I_out_array,
            'V_middle': V_middle_array
        }
    

if __name__ == "__main__":
    weight_matrix = np.array([
        [1.0, 0.3, 0.1, 0.0],
        [0.2, 1.0, 0.4, 0.1],
        [0.1, 0.3, 1.0, 0.3],
        [0.0, 0.1, 0.2, 1.0]
    ])
    sim = VMMWTA(n_inputs=4,n_outputs=4,weight_matrix=weight_matrix)
    
    def V_d_input(t):
        amp = 0.005
        
        return np.array([
            0.08 + amp * np.sin(2 * np.pi * 100 * t),
            0.10 + amp * np.sin(2 * np.pi * 200 * t),
            0.12 + amp * np.sin(2 * np.pi * 300 * t),
            0.14 + amp * np.sin(2 * np.pi * 400 * t)
        ])
    
    sim.I_prog = np.ones(4) * 5e-9 
    
    t, results = sim.simulate(V_d_input, t_span=(0, 30e-3), dt=5e-7)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].set_title('V_d')
    for i in range(sim.n):
        freq = [100, 200, 300, 400][i]
        axes[0,0].plot(t * 1e3, results['V_d_input'][:, i], label=f'V_d[{i}] ({freq}Hz)', linewidth=1.5)
    axes[0,0].set_ylabel('Voltage (V)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_title('V_out')
    for i in range(sim.n):
        axes[0,1].plot(t * 1e3, results['V_out'][:, i], label=f'V_out[{i}]', linewidth=2)
    axes[0,1].set_ylabel('Voltage (V)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].set_title('I_out')
    for i in range(sim.n):
        axes[1,0].plot(t * 1e3, results['I_out'][:, i] * 1e9, label=f'I_out[{i}]', linewidth=1.5)
    axes[1,0].set_xlabel('Time (ms)')
    axes[1,0].set_ylabel('Current')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].set_title('Extended Diff Pair Voltage (V)')
    axes[1,1].plot(t * 1e3, results['V_middle'], linewidth=2, color='purple', label='V (middle node)')
    axes[1,1].set_xlabel('Time (ms)')
    axes[1,1].set_ylabel('Voltage (V)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

   
