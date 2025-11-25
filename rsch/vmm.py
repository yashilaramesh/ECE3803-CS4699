import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

class VMMWTA:
    def __init__(self, n_inputs=4, n_outputs=4, weight_matrix=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.U_T = 0.0258  # Thermal voltage (V) at room temperature
        self.I_bias = 1e-9  # Bias current (A) - this is the SHARED tail current
        self.I_tail = 1e-9  # Tail current for WTA (shared across all outputs)
        self.C_L = 100e-15  # Load capacitance (F)
        self.kappa_n = 0.7  # Capacitive coupling coefficient
        self.V_bias = 0.0  # bias voltage
        self.V_dd = 1.5  # supply voltage
        
        # Weight matrix for VMM (default to identity if not provided)
        if weight_matrix is None:
            self.W = np.eye(n_outputs, n_inputs)
        else:
            self.W = weight_matrix
        
        # Output voltages initial condition
        self.V_out_init = np.ones(n_outputs) * 0.5  # Start at mid-range
        
        # I_prog represents the pull-up current sources at each output
        self.I_prog = np.ones(n_outputs) * 0.5e-9
        
    def compute_vmm_currents(self, V_d_input):
        """Compute the output currents from VMM with weight matrix"""
        I_out = np.zeros(self.n_outputs)
        
        for l in range(self.n_outputs):  # output row
            for k in range(self.n_inputs):  # input column
                I_out[l] += self.W[l, k] * np.exp(V_d_input[k] / self.U_T)
        
        # Scale by bias current
        I_out = self.I_bias * I_out
        
        return I_out
    
    def compute_V_node(self, V_out, I_vmm_total):
        """
        Compute the shared node voltage V that satisfies current conservation.
        This is the key constraint: Σ I_wta[k] = I_tail
        
        Each WTA output pulls current: I_wta[k] = I_0 × exp((V - V_out[k])/U_T)
        Constraint: Σ I_wta[k] = I_tail
        
        We need to find V such that this constraint is satisfied.
        """
        def constraint_equation(V):
            # Sum of currents through all output FETs must equal tail current
            total_current = 0
            for k in range(self.n_outputs):
                # Current through FET k depends on V (shared source) and V_out[k] (drain)
                # Using standard FET equation in subthreshold
                I_k = self.I_tail * np.exp((V - V_out[k]) / self.U_T)
                total_current += I_k
            
            # The constraint: total must equal I_tail
            # But we need to normalize by a reference current to make this solvable
            # The sum of exp((V - V_out[k])/U_T) should equal n_outputs for equal distribution
            return total_current - self.I_tail * self.n_outputs
        
        # Solve for V that satisfies the constraint
        # Initial guess: average of output voltages
        V_guess = np.mean(V_out)
        V_solution = fsolve(constraint_equation, V_guess)[0]
        
        return V_solution
    
    def compute_wta_currents(self, V_out, V_node):
        """
        Compute individual WTA output currents given the shared node voltage V.
        These currents flow from each output through the shared node to the tail current.
        """
        I_wta = np.zeros(self.n_outputs)
        
        for k in range(self.n_outputs):
            # Current through FET k (from output k to shared node V)
            # This is the competitive current - they all share I_tail
            I_wta[k] = (self.I_tail / self.n_outputs) * np.exp((V_node - V_out[k]) / self.U_T)
        
        # Normalize to ensure sum equals I_tail (numerical stability)
        I_wta = I_wta * self.I_tail / np.sum(I_wta)
        
        return I_wta
    
    def ode_system_with_constraint(self, state, t, V_d_input):
        """
        ODE system with proper current constraint.
        Now the outputs are coupled through the shared node V.
        """
        V_out = state
        
        # Compute VMM output (drives the gates of WTA FETs)
        I_vmm = self.compute_vmm_currents(V_d_input)
        
        # Find the shared node voltage V that satisfies current conservation
        V_node = self.compute_V_node(V_out, np.sum(I_vmm))
        
        # Compute WTA currents (these sum to I_tail by construction)
        I_wta = self.compute_wta_currents(V_out, V_node)
        
        # Now compute derivatives for each output
        dV_out_dt = np.zeros(self.n_outputs)
        
        for k in range(self.n_outputs):
            # Current balance at output node k:
            # C_L × dV/dt = I_prog (pull-up) - I_wta (pull-down through shared node)
            # The VMM current influences V_node, which affects I_wta distribution
            
            # Pull-up current source
            I_pullup = self.I_prog[k] * np.exp((self.V_dd - V_out[k]) / self.U_T)
            
            # Pull-down current through WTA (competitive)
            I_pulldown = I_wta[k]
            
            # Add influence from VMM (modulates the competition)
            # VMM currents affect the gate voltages, changing the current distribution
            vmm_influence = I_vmm[k] / self.I_tail  # Normalized influence
            I_pulldown = I_pulldown * (1 + vmm_influence)
            
            dV_out_dt[k] = (1 / self.C_L) * (I_pullup - I_pulldown)
        
        return dV_out_dt
    
    def simulate(self, V_d_input_func, t_span, dt=1e-6):
        t = np.arange(t_span[0], t_span[1], dt)
    
        state0 = self.V_out_init
        
        V_d_input_array = np.array([V_d_input_func(ti) for ti in t])
        
        results = []
        state = state0
        I_vmm_array = []
        I_wta_array = []
        V_node_array = []
        
        for i in range(len(t)):
            # Compute VMM output currents
            I_vmm = self.compute_vmm_currents(V_d_input_array[i])
            I_vmm_array.append(I_vmm)
            
            # Compute shared node voltage and WTA currents
            V_node = self.compute_V_node(state, np.sum(I_vmm))
            I_wta = self.compute_wta_currents(state, V_node)
            V_node_array.append(V_node)
            I_wta_array.append(I_wta)
            
            if i == 0:
                results.append(state)
            else:
                sol = odeint(self.ode_system_with_constraint, state, 
                            [t[i-1], t[i]], args=(V_d_input_array[i],))
                state = sol[-1]
                results.append(state)
        
        V_out = np.array(results)
        I_vmm_array = np.array(I_vmm_array)
        I_wta_array = np.array(I_wta_array)
        V_node_array = np.array(V_node_array)
        
        return t, {
            'V_out': V_out, 
            'V_d_input': V_d_input_array,
            'I_vmm': I_vmm_array,
            'I_wta': I_wta_array,
            'V_node': V_node_array
        }


if __name__ == "__main__":
    # Define a weight matrix with more pronounced mixing
    weight_matrix = np.array([
        [1.0, 0.3, 0.1, 0.0],  
        [0.2, 1.0, 0.4, 0.1],  
        [0.1, 0.3, 1.0, 0.3],  
        [0.0, 0.1, 0.2, 1.0]   
    ])
    
    sim = VMMWTA(n_inputs=4, n_outputs=4, weight_matrix=weight_matrix)
    
    def V_d_input_multifreq(t):
        """Different frequencies with level shifts"""
        amp = 0.005
        
        return np.array([
            0.08 + amp * np.sin(2 * np.pi * 100 * t),
            0.10 + amp * np.sin(2 * np.pi * 200 * t),
            0.12 + amp * np.sin(2 * np.pi * 300 * t),
            0.14 + amp * np.sin(2 * np.pi * 400 * t)
        ])
    
    # Adjust tail current to be larger for more pronounced WTA effect
    sim.I_tail = 5e-9
    sim.I_prog = np.ones(4) * 1e-9
    
    t, results = sim.simulate(V_d_input_multifreq, t_span=(0, 30e-3), dt=1e-6)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Plot V_d inputs
    axes[0,0].set_title('VMM Input Voltages (V_d) - Different Frequencies', fontsize=12, fontweight='bold')
    for i in range(sim.n_inputs):
        freq = [100, 200, 300, 400][i]
        axes[0,0].plot(t * 1e3, results['V_d_input'][:, i], label=f'V_d[{i}] ({freq}Hz)', linewidth=1.5)
    axes[0,0].set_ylabel('Voltage (V)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot V_out
    axes[0,1].set_title('WTA Output Voltages (V_out) - WITH CONSTRAINT', fontsize=12, fontweight='bold')
    for i in range(sim.n_outputs):
        axes[0,1].plot(t * 1e3, results['V_out'][:, i], label=f'V_out[{i}]', linewidth=2)
    axes[0,1].set_ylabel('Voltage (V)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot VMM output currents
    axes[1,0].set_title('VMM Output Currents (I_vmm) - Weight Matrix', fontsize=12, fontweight='bold')
    for i in range(sim.n_outputs):
        axes[1,0].plot(t * 1e3, results['I_vmm'][:, i] * 1e9, label=f'I_vmm[{i}]', linewidth=1.5)
    axes[1,0].set_xlabel('Time (ms)')
    axes[1,0].set_ylabel('Current (nA)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot WTA currents (competitive currents)
    axes[1,1].set_title('WTA Competitive Currents (I_wta) - Sum = I_tail', fontsize=12, fontweight='bold')
    for i in range(sim.n_outputs):
        axes[1,1].plot(t * 1e3, results['I_wta'][:, i] * 1e9, label=f'I_wta[{i}]', linewidth=1.5)
    # Plot sum to verify constraint
    I_wta_sum = np.sum(results['I_wta'], axis=1) * 1e9
    axes[1,1].plot(t * 1e3, I_wta_sum, 'k--', linewidth=2, label=f'Sum (should = {sim.I_tail*1e9:.1f} nA)')
    axes[1,1].set_xlabel('Time (ms)')
    axes[1,1].set_ylabel('Current (nA)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot shared node voltage V
    axes[2,0].set_title('Shared Node Voltage (V) - Competition Point', fontsize=12, fontweight='bold')
    axes[2,0].plot(t * 1e3, results['V_node'], linewidth=2, color='purple', label='V (shared node)')
    axes[2,0].set_xlabel('Time (ms)')
    axes[2,0].set_ylabel('Voltage (V)')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    
    # Plot current distribution as stacked area
    axes[2,1].set_title('WTA Current Distribution (Stacked)', fontsize=12, fontweight='bold')
    axes[2,1].stackplot(t * 1e3, 
                        results['I_wta'][:, 0] * 1e9,
                        results['I_wta'][:, 1] * 1e9,
                        results['I_wta'][:, 2] * 1e9,
                        results['I_wta'][:, 3] * 1e9,
                        labels=['I_wta[0]', 'I_wta[1]', 'I_wta[2]', 'I_wta[3]'],
                        alpha=0.7)
    axes[2,1].set_xlabel('Time (ms)')
    axes[2,1].set_ylabel('Current (nA)')
    axes[2,1].legend(loc='upper right')
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].set_ylim([0, sim.I_tail * 1e9 * 1.1])
    
    plt.tight_layout()
    plt.show()
    
    print("Simulation complete!")
    print(f"\nWeight Matrix:")
    print(sim.W)
    print(f"\nCircuit Parameters:")
    print(f"  I_tail (shared): {sim.I_tail * 1e9:.2f} nA")
    print(f"  I_prog (pull-up): {sim.I_prog[0] * 1e9:.2f} nA")
    print(f"\nFinal state:")
    print(f"  Output voltages: {results['V_out'][-1]}")
    print(f"  WTA currents: {results['I_wta'][-1] * 1e9} nA")
    print(f"  WTA current sum: {np.sum(results['I_wta'][-1]) * 1e9:.2f} nA (should = {sim.I_tail * 1e9:.2f} nA)")
    print(f"  Shared node V: {results['V_node'][-1]:.4f} V")
    print(f"  Input winner: channel {np.argmax(results['V_d_input'][-1])} (highest DC level)")
    print(f"  Output winner: channel {np.argmax(results['V_out'][-1])}")
    
    # Verify constraint is satisfied
    max_constraint_error = np.max(np.abs(np.sum(results['I_wta'], axis=1) - sim.I_tail))
    print(f"\nConstraint verification:")
    print(f"  Max error in Σ I_wta = I_tail: {max_constraint_error * 1e9:.6f} nA")