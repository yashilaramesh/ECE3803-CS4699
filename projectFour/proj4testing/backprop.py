import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.integrate import odeint

class AudioPredictionNetworkODE:
    """
    Two-layer neural network with ODE-based weight updates
    Uses multiple timescales: fast for signals, slow for weights
    """
    def __init__(self, n_inputs=64, n_hidden=20, n_outputs=64, 
                 hidden_gain=3.0, use_output_sigmoid=False):
        """
        n_inputs: number of input samples (window size)
        n_hidden: number of hidden layer neurons
        n_outputs: number of output samples
        hidden_gain: gain for hidden layer sigmoids (2-4 recommended)
        use_output_sigmoid: False for linear output (recommended for audio)
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.hidden_gain = hidden_gain
        self.use_output_sigmoid = use_output_sigmoid
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(n_hidden, n_inputs) * 0.01
        self.W2 = np.random.randn(n_outputs, n_hidden) * 0.01
        
        # Weight velocities (for ODE integration)
        self.dW1_dt = np.zeros_like(self.W1)
        self.dW2_dt = np.zeros_like(self.W2)
        
        # For storing during forward pass
        self.x = None
        self.hidden = None
        self.output = None
        
        # Timescale parameters
        self.tau_fast = 1.0      # Fast timescale for signals
        self.tau_slow = 100.0    # Slow timescale for weights (100x slower)
        
    def sigmoid(self, z):
        """Sigmoid activation with gain"""
        return 1 / (1 + np.exp(-self.hidden_gain * np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, sigmoid_output):
        """Derivative of sigmoid"""
        return self.hidden_gain * sigmoid_output * (1 - sigmoid_output)
    
    def forward(self, x):
        """
        Forward pass through network (fast timescale)
        x: input vector (n_inputs,)
        returns: output vector (n_outputs,)
        """
        self.x = x
        
        # Hidden layer (with sigmoid)
        z1 = self.W1 @ x
        self.hidden = self.sigmoid(z1)
        
        # Output layer (linear or sigmoid)
        z2 = self.W2 @ self.hidden
        if self.use_output_sigmoid:
            self.output = self.sigmoid(z2)
        else:
            self.output = z2  # Linear output for audio
            
        return self.output
    
    def compute_weight_derivatives(self, target, learning_rate=0.001):
        """
        Compute dW/dt for ODE formulation
        
        The weight update ODEs:
        dW2/dt = -η/τ_slow * ∂E/∂W2
        dW1/dt = -η/τ_slow * ∂E/∂W1
        
        Where E = 1/2 * ||target - output||²
        """
        # Compute output error
        output_error = target - self.output
        
        # Output layer gradient
        if self.use_output_sigmoid:
            output_delta = output_error * self.sigmoid_derivative(self.output)
        else:
            output_delta = output_error
        
        # Hidden layer gradient (backpropagate)
        hidden_error = self.W2.T @ output_delta
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)
        
        # Compute weight derivatives (ODE right-hand side)
        # Negative gradient descent: dW/dt = +η * ∂E/∂W (since error = target - output)
        self.dW2_dt = (learning_rate / self.tau_slow) * np.outer(output_delta, self.hidden)
        self.dW1_dt = (learning_rate / self.tau_slow) * np.outer(hidden_delta, self.x)
        
        return np.mean(output_error ** 2)
    
    def euler_update(self, dt=1.0):
        """
        Euler integration step for weight updates
        W(t + dt) = W(t) + dW/dt * dt
        
        dt: timestep (usually 1.0 for one training step)
        """
        self.W2 += self.dW2_dt * dt
        self.W1 += self.dW1_dt * dt
    
    def rk4_update(self, target, learning_rate, dt=1.0):
        """
        4th-order Runge-Kutta integration (more accurate than Euler)
        This is optional but provides better ODE integration
        """
        # Save current weights
        W1_old = self.W1.copy()
        W2_old = self.W2.copy()
        
        # k1
        self.compute_weight_derivatives(target, learning_rate)
        k1_W1 = self.dW1_dt
        k1_W2 = self.dW2_dt
        
        # k2
        self.W1 = W1_old + 0.5 * dt * k1_W1
        self.W2 = W2_old + 0.5 * dt * k1_W2
        self.forward(self.x)
        self.compute_weight_derivatives(target, learning_rate)
        k2_W1 = self.dW1_dt
        k2_W2 = self.dW2_dt
        
        # k3
        self.W1 = W1_old + 0.5 * dt * k2_W1
        self.W2 = W2_old + 0.5 * dt * k2_W2
        self.forward(self.x)
        self.compute_weight_derivatives(target, learning_rate)
        k3_W1 = self.dW1_dt
        k3_W2 = self.dW2_dt
        
        # k4
        self.W1 = W1_old + dt * k3_W1
        self.W2 = W2_old + dt * k3_W2
        self.forward(self.x)
        self.compute_weight_derivatives(target, learning_rate)
        k4_W1 = self.dW1_dt
        k4_W2 = self.dW2_dt
        
        # Final update
        self.W1 = W1_old + (dt/6.0) * (k1_W1 + 2*k2_W1 + 2*k3_W1 + k4_W1)
        self.W2 = W2_old + (dt/6.0) * (k1_W2 + 2*k2_W2 + 2*k3_W2 + k4_W2)
        
        # Return error
        self.forward(self.x)
        return np.mean((target - self.output) ** 2)
    
    def train_step_ode(self, x, target, learning_rate=0.001, 
                       dt=1.0, use_rk4=False):
        """
        Single training step using ODE formulation
        
        x: input
        target: desired output
        learning_rate: η in the ODE
        dt: integration timestep
        use_rk4: if True, use RK4; if False, use Euler
        """
        # Forward pass (fast timescale)
        self.forward(x)
        
        # Compute weight derivatives (slow timescale)
        error = self.compute_weight_derivatives(target, learning_rate)
        
        # Integrate ODEs
        if use_rk4:
            error = self.rk4_update(target, learning_rate, dt)
        else:
            self.euler_update(dt)
        
        return error
    
    def freeze_hidden_layer(self):
        """Store hidden layer weights for later restoration"""
        self.W1_frozen = self.W1.copy()
    
    def train_output_only_ode(self, target, learning_rate=0.001, dt=1.0):
        """Train only output layer (W2), keeping W1 frozen"""
        output_error = target - self.output
        
        if self.use_output_sigmoid:
            output_delta = output_error * self.sigmoid_derivative(self.output)
        else:
            output_delta = output_error
        
        # Only compute dW2/dt
        self.dW2_dt = (learning_rate / self.tau_slow) * np.outer(output_delta, self.hidden)
        self.dW1_dt = np.zeros_like(self.W1)  # Frozen
        
        # Update only W2
        self.W2 += self.dW2_dt * dt
        
        return np.mean(output_error ** 2)


def load_and_preprocess_audio(filepath, target_sr=16000, normalize=True):
    audio = None
    sr = target_sr
    
    try:
        sr, audio = wavfile.read(filepath)
        print(f"✓ Loaded with scipy.wavfile")
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0
        else:
            audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
    except Exception as e:
        print(f"✗ scipy.wavfile failed: {e}")
        
        try:
            import soundfile as sf
            audio, sr = sf.read(filepath, dtype='float32')
            print(f"✓ Loaded with soundfile")
            
            # If stereo, convert to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                
        except ImportError:
            print("✗ soundfile library not installed")
            print("  Install with: pip install soundfile")
            
        except Exception as e2:
            print(f"✗ soundfile failed: {e2}")
    
    # If all loading methods failed, generate synthetic audio
    if audio is None:
        print("\n" + "!" * 60)
        print("WARNING: Could not load audio file")
        print("Generating synthetic audio for demonstration...\n")
        
        # Generate synthetic audio (mixture of sine waves)
        sr = target_sr
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sr * duration))
        audio = (0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
                 0.2 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
                 0.15 * np.sin(2 * np.pi * 659.25 * t))  # E5
        audio = audio / np.max(np.abs(audio))
    
    else:
        # Resample if needed
        if sr != target_sr:
            print(f"Resampling from {sr} Hz to {target_sr} Hz...")
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)
            sr = target_sr
        
        # Normalize
        if normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    return audio, sr


def create_sliding_windows(audio, window_size=64, prediction_offset=0):
    """
    Create sliding window inputs and targets
    prediction_offset: 0 for reproduction, >0 for prediction
    """
    X = []
    y = []
    
    for i in range(len(audio) - window_size - prediction_offset):
        X.append(audio[i:i + window_size])
        
        if prediction_offset == 0:
            y.append(audio[i:i + window_size])
        else:
            y.append(audio[i + window_size:i + window_size + prediction_offset])
    
    return np.array(X), np.array(y)


def compute_covariance_analysis(X):
    """
    Compute covariance matrix and eigenvalue spread
    """
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    spread = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
    
    return cov_matrix, eigenvalues, spread


def train_network_ode(network, X_train, y_train, epochs=10, learning_rate=0.001,
                      dt=1.0, use_rk4=False, train_output_only=False):
    """
    Train the network using ODE formulation
    """
    errors_per_epoch = []
    
    for epoch in range(epochs):
        epoch_errors = []
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        
        for idx in indices:
            x = X_train[idx]
            target = y_train[idx]
            
            # Forward pass
            network.forward(x)
            
            # ODE-based weight update
            if train_output_only:
                error = network.train_output_only_ode(target, learning_rate, dt)
            else:
                error = network.train_step_ode(x, target, learning_rate, dt, use_rk4)
            
            epoch_errors.append(error)
        
        avg_error = np.mean(epoch_errors)
        errors_per_epoch.append(avg_error)
        
        print(f"Epoch {epoch + 1}/{epochs}, Average MSE: {avg_error:.6f}")
    
    return errors_per_epoch


# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    print("=" * 60)
    print("Audio Self-Supervised Learning Network (ODE Formulation)")
    print("=" * 60)
    
    # Load audio
    audio_file = "/Users/yashila/Documents/GitHub/ECE3803-CS4699/projectFour/ironic.wav"
    print(f"\nLoading audio from: {audio_file}")
    audio, sr = load_and_preprocess_audio(audio_file)
    print(f"Audio length: {len(audio)} samples ({len(audio)/sr:.2f} seconds)")
    print(f"Sample rate: {sr} Hz")
    
    # Task 1: REPRODUCTION
    print("\n" + "=" * 60)
    print("TASK 1: Audio Reproduction (ODE-based training)")
    print("=" * 60)
    
    window_size = 64
    X_train, y_train = create_sliding_windows(audio, window_size, prediction_offset=0)
    print(f"Training samples: {len(X_train)}")
    
    # Covariance analysis
    print("\nCovariance Analysis:")
    cov_matrix, eigenvalues, spread = compute_covariance_analysis(X_train)
    print(f"Eigenvalue spread (λ_max/λ_min): {spread:.2e}")
    print(f"Top 5 eigenvalues: {eigenvalues[:5]}")
    print(f"Bottom 5 eigenvalues: {eigenvalues[-5:]}")
    print(f"\nTimescale separation:")
    print(f"  τ_fast (signals): 1.0")
    print(f"  τ_slow (weights): 100.0")
    print(f"  Ratio τ_slow/τ_fast: 100x")
    
    # Create network
    n_hidden = 1
    network_repro = AudioPredictionNetworkODE(
        n_inputs=64, 
        n_hidden=n_hidden, 
        n_outputs=64,
        hidden_gain=3.0,
        use_output_sigmoid=False
    )
    
    print(f"\nNetwork architecture: {64} → {n_hidden} → {64}")
    print("ODE: dW/dt = (η/τ_slow) * ∂E/∂W")
    print("Training with Euler integration...")
    
    errors_repro = train_network_ode(
        network_repro, X_train, y_train, 
        epochs=20, 
        learning_rate=0.001,
        dt=1.0,
        use_rk4=False  # Set True for RK4 integration
    )
    
    # Task 2: PREDICTION
    print("\n" + "=" * 60)
    print("TASK 2: Audio Prediction (8 samples ahead)")
    print("=" * 60)
    
    prediction_samples = 8
    X_train_pred, y_train_pred = create_sliding_windows(
        audio, window_size, prediction_offset=prediction_samples
    )
    
    # Create network
    network_pred = AudioPredictionNetworkODE(
        n_inputs=64, 
        n_hidden=n_hidden,
        n_outputs=prediction_samples,
        hidden_gain=2.0,
        use_output_sigmoid=False
    )
    
    # Copy hidden layer weights
    network_pred.W1 = network_repro.W1.copy()
    network_pred.freeze_hidden_layer()
    
    print("Stage 1: Training output layer only (frozen hidden layer)...")
    
    errors_pred_stage1 = train_network_ode(
        network_pred, X_train_pred, y_train_pred,
        epochs=20,
        learning_rate=0.001,
        dt=1.0,
        train_output_only=True
    )
    
    # Stage 2: Train full network
    network_pred.W1 = network_pred.W1_frozen.copy()
    
    print("\nStage 2: Training full network...")
    errors_pred_stage2 = train_network_ode(
        network_pred, X_train_pred, y_train_pred,
        epochs=20,
        learning_rate=0.0005,
        dt=1.0
    )
    
    errors_pred_full = errors_pred_stage1 + errors_pred_stage2
    
    # ============= GENERATE PREDICTED AUDIO =============
    print("\n" + "=" * 60)
    print("Generating Predicted Audio")
    print("=" * 60)
    
    predicted_audio = []
    for i in range(len(audio) - window_size - prediction_samples):
        input_window = audio[i:i + window_size]
        network_pred.forward(input_window)
        predicted_audio.append(network_pred.output)
    
    predicted_audio = np.array(predicted_audio)
    predicted_continuous = predicted_audio[:, 0]
    
    print(f"Generated {len(predicted_continuous)} predicted samples")
    
    # Get the corresponding original audio (aligned with predictions)
    original_aligned = audio[window_size + prediction_samples:window_size + prediction_samples + len(predicted_continuous)]
    
    # Save audio files
    output_filename = "predicted_ep20.wav"
    predicted_int16 = np.int16(predicted_continuous * 32767)
    wavfile.write(output_filename, sr, predicted_int16)
    print(f"Predicted audio saved as '{output_filename}'")
    
    original_filename = "original_audio_snippet_ode_ep20.wav"
    original_int16 = np.int16(original_aligned * 32767)
    wavfile.write(original_filename, sr, original_int16)
    print(f"Original audio snippet saved as '{original_filename}'")
    
    # ============= PLOTTING =============
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Plot 1: Reproduction error
    axes[0, 0].semilogy(errors_repro, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Average MSE (log scale)')
    axes[0, 0].set_title('Task 1: Reproduction (ODE Training)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction error
    axes[0, 1].semilogy(range(len(errors_pred_full)), errors_pred_full, 'r-', linewidth=2)
    axes[0, 1].axvline(x=20, color='k', linestyle='--', label='Stage 1→2')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Average MSE (log scale)')
    axes[0, 1].set_title('Task 2: Prediction (ODE Training)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Eigenvalue spectrum
    axes[1, 0].semilogy(eigenvalues, 'go-', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Eigenvalue Index')
    axes[1, 0].set_ylabel('Eigenvalue (log scale)')
    axes[1, 0].set_title(f'Covariance Eigenspectrum (spread: {spread:.2e})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Audio waveform (first 5000 samples)
    time_axis = np.arange(min(5000, len(audio))) / sr
    axes[1, 1].plot(time_axis, audio[:len(time_axis)], 'purple', linewidth=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Audio Waveform (first 5000 samples)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # NEW PLOT 5: Prediction vs Original Overlay (short segment)
    seconds_to_plot = 0.5  # 500ms
    samples_to_plot = int(seconds_to_plot * sr)
    samples_to_plot = min(samples_to_plot, len(predicted_continuous))
    
    time_axis_pred = np.arange(samples_to_plot) / sr
    
    axes[2, 0].plot(time_axis_pred, original_aligned[:samples_to_plot], 'b-', 
                    linewidth=1.5, alpha=0.7, label='Original (Truth)')
    axes[2, 0].plot(time_axis_pred, predicted_continuous[:samples_to_plot], 'r--', 
                    linewidth=1.5, alpha=0.7, label='Predicted (8 steps ahead)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].set_title(f'Prediction vs Original (first {seconds_to_plot}s)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # NEW PLOT 6: Prediction Error Over Time
    prediction_error = np.abs(original_aligned - predicted_continuous)
    error_samples = min(10000, len(prediction_error))
    time_axis_error = np.arange(error_samples) / sr
    
    axes[2, 1].plot(time_axis_error, prediction_error[:error_samples], 'orange', linewidth=0.5)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Absolute Error')
    axes[2, 1].set_title('Prediction Error Over Time')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('audio_network_results_ode_ep20.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'audio_network_results_ode_ep20.png'")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final reproduction error: {errors_repro[-1]:.6f}")
    print(f"Final prediction error: {errors_pred_full[-1]:.6f}")
    print(f"Hidden layer nodes used: {n_hidden}")
    print(f"\nODE Integration Method: Euler (dt=1.0)")
    print(f"Timescale ratio: τ_slow/τ_fast = 100")
    
    # Calculate alignment metrics
    mse_aligned = np.mean((original_aligned - predicted_continuous) ** 2)
    correlation = np.corrcoef(original_aligned, predicted_continuous)[0, 1]
    
    print(f"\n=== Prediction Quality Metrics ===")
    print(f"Aligned MSE: {mse_aligned:.6e}")
    print(f"Correlation coefficient: {correlation:.4f}")
    
    print(f"\nOutput files created:")
    print(f"  - {output_filename}")
    print(f"  - {original_filename}")
    print(f"  - audio_network_results_ode_ep20.png")