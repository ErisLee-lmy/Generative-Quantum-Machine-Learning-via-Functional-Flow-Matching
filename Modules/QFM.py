# %%
import pennylane as qml
from pennylane import numpy as np

class QuantumFlowTomography:
    """
    Quantum Flow Matching inspired quantum process tomography via PennyLane.

    Args:
        n_qubits_in (int): number of input qubits.
        n_qubits_out (int): number of output qubits.
        depth (int): number of HEA layers per time-step.
        timesteps (int): number of time steps in flow (>=1). The procedure will train a small HEA for each step.
        shots (int): number of shots for device. NOTE: this implementation uses analytic density outputs (shots=None)
                     for density-based loss. If shots>0, sampling-based tomography is NOT implemented here and
                     analytic device will be used instead (see notes in class docstring).
        backend (str): PennyLane device name, e.g. 'default.qubit'.

    Data expectations for fit():
        samples_input: np.ndarray shape (N, 2**n_qubits_in) - complex state vectors (amplitudes) for input pure states.
        samples_output: np.ndarray shape (N, 2**n_qubits_out) - complex state vectors for output states after the true process.
    """

    def __init__(self, n_qubits_in, n_qubits_out, depth=2, timesteps=4, shots=0, backend="default.qubit"):
        self.n_in = n_qubits_in
        self.n_out = n_qubits_out
        self.ancilla = n_qubits_out  # Stinespring ancilla qubits
        self.n_total = self.n_in + self.ancilla  # total qubits for unitary
        
        self.depth = depth
        self.timesteps = timesteps
        self.shots = shots
        self.backend = backend

        # number of wires for the full unitary (system_in wires + ancilla wires)
        self.sys_wires = list(range(self.n_in))
        self.anc_wires = list(range(self.n_in, self.n_in + self.ancilla))
        self.total_wires = list(range(self.n_in + self.ancilla))

        # device: for density output we require analytic device (shots=None)
        if self.shots and self.shots > 0:
            # NOTE: sampling-based tomography is not implemented; use analytic device for training.
            # We keep backend name but instantiate device with analytic mode.
            print("Warning: shots>0 was provided, but this implementation uses analytic density outputs."
                  " Training will proceed with analytic device (shots=None).")
        self.dev = qml.device(backend, wires=self.total_wires)

        # initialize parameters per timestep: each timestep has params for HEA layers
        # shape: (timesteps, depth, n_params_per_layer)
        # We'll parametrize each layer by rx, rz per wire (2 params per wire)
        self.n_params_per_layer = 2 * self.n_total
        # params: list length timesteps, each is a (depth, n_params_per_layer) array
        self.params = [np.random.normal(0, 0.1, (self.depth, self.n_params_per_layer), requires_grad=True)
                       for _ in range(self.timesteps)]

        # Build qnode generator for a single timestep unitary that returns density on system wires
        # We'll create a closure that builds QNode given theta and input_state
        def make_qnode():
            @qml.qnode(self.dev, interface="autograd")
            def qnode(theta, state_vec):
                # state_vec: complex vector length 2**n_in
                # Prepare input on system wires:
                qml.templates.MottonenStatePreparation(state_vec, wires=self.sys_wires)
                # Initialize ancilla to |0...0> automatically

                # Apply parametrized HEA acting on system + ancilla
                # theta shape expected: (depth, n_params_per_layer)
                self._apply_heas(theta)
                # return reduced density matrix on system wires
                return qml.density_matrix(self.sys_wires)
            return qnode

        self._make_qnode = make_qnode
        # a qnode instance will be created on-demand in fit (so device compilation respects params)
        self.qnodes = [None] * self.timesteps

    def _apply_heas(self, theta):
        """
        Apply HEA layers according to theta (depth * n_params_per_layer).
        Each layer: single-qubit rotations Rx, Rz on each wire followed by chain of CNOT entanglers.
        """
        depth = theta.shape[0]
        nparams = theta.shape[1]
        assert nparams == self.n_params_per_layer
        for d in range(depth):
            layer_params = theta[d]
            # layer_params split into pairs (rx, rz) per wire
            for w_idx, wire in enumerate(self.total_wires):
                rx_param = layer_params[2 * w_idx]
                rz_param = layer_params[2 * w_idx + 1]
                qml.RX(rx_param, wires=wire)
                qml.RZ(rz_param, wires=wire)
            # simple entangler: chain CNOTs
            for w in range(self.n_total - 1):
                qml.CNOT(wires=[w, w + 1])
            # also close the ring
            if self.n_total > 1:
                qml.CNOT(wires=[self.n_total - 1, 0])

    def default_interp(self, s, rho_in_pure, rho_out):
        """
        Default linear interpolation for a given s in [0,1]:
            rho_t = (1-s) * rho_in_pure + s * rho_out
        where rho_in_pure is |psi><psi| and rho_out is the sample output density matrix.
        """
        return (1.0 - s) * rho_in_pure + s * rho_out

    def _ensure_qnode(self, step_idx):
        """Create a qnode for given step if not present."""
        if self.qnodes[step_idx] is None:
            self.qnodes[step_idx] = self._make_qnode()

    def fit(self, samples_input, samples_output, epochs_per_step=200, lr=0.05, interp_fn=None,
            batch_size=None, verbose=True):
        """
        Train the sequence of small HEAs across timesteps.
        """
        if interp_fn is None:
            interp_fn = self.default_interp

        N = samples_input.shape[0]
        assert samples_input.shape[0] == samples_output.shape[0], "Input/output sample counts must match."

        # Precompute density matrices
        rho_in_pure_list = []
        rho_out_list = []
        for i in range(N):
            psi_in = np.array(samples_input[i], dtype=np.complex128)
            psi_out = np.array(samples_output[i], dtype=np.complex128)
            rho_in = np.outer(psi_in, np.conjugate(psi_in))
            rho_out = np.outer(psi_out, np.conjugate(psi_out))

            # handle dim mismatch (best-effort padding/truncation)
            if self.n_in != self.n_out:
                if rho_out.shape[0] < rho_in.shape[0]:
                    pad = rho_in.shape[0] // rho_out.shape[0]
                    rho_out = np.kron(rho_out, np.eye(pad) / pad)
                elif rho_out.shape[0] > rho_in.shape[0]:
                    rho_out = rho_out[:rho_in.shape[0], :rho_in.shape[0]]

            rho_in_pure_list.append(rho_in)
            rho_out_list.append(rho_out)

        # Training timesteps sequentially
        for step in range(self.timesteps):
            if verbose:
                print(f"\n=== Training timestep {step+1}/{self.timesteps} ===")

            # ensure qnode for this step
            self._ensure_qnode(step)
            qnode = self.qnodes[step]

            theta = self.params[step]
            opt = qml.AdamOptimizer(stepsize=lr)

            # cost function for this step
            def cost_fn(theta_flat, batch_indices):
                theta_arr = theta_flat.reshape(theta.shape)
                loss = 0.0
                for idx in batch_indices:
                    psi_in = samples_input[idx]
                    rho_theta = qnode(theta_arr, psi_in)
                    s = (step + 1) / float(self.timesteps)
                    rho_t = interp_fn(s, rho_in_pure_list[idx], rho_out_list[idx])
                    overlap = np.real(np.trace(np.matmul(rho_t, rho_theta)))
                    loss += (1.0 - overlap)
                return loss / len(batch_indices)

            # training loop for this timestep
            theta_flat = theta.reshape(-1)
            for epoch in range(epochs_per_step):
                # sample batch indices
                if batch_size is None or batch_size >= N:
                    batch_idx = list(range(N))
                else:
                    batch_idx = np.random.choice(N, batch_size, replace=False)

                # update parameters
                theta_flat, current_loss = opt.step_and_cost(
                    lambda t: cost_fn(t, batch_idx), theta_flat
                )

                if epoch % max(1, epochs_per_step // 5) == 0 and verbose:
                    print(f" Step {step+1} Epoch {epoch}/{epochs_per_step}  loss={current_loss:.6f}")

            # save updated params
            self.params[step] = theta_flat.reshape(theta.shape)
            if verbose:
                print(f"Finished timestep {step+1}. Saved params.\n")


    def predict_density(self, input_state, compose_all_steps=True):
        """
        Given an input state vector (length 2**n_in), return the predicted output density matrix.
        If compose_all_steps=True, we apply sequentially all learned unitaries (i.e. U_step1, then U_step2,...)
        by simulating the forward action on the state/ancilla. For simplicity we re-run the QNodes sequentially
        on the evolving state (this is a simulated composition).
        """
        psi = np.array(input_state, dtype=np.complex128)

        # sequentially apply each trained U (we reuse qnodes, but as QNode expects to start with input on system,
        # we must re-prepare the full state: system=psi, ancilla=|0>, then apply current step unitary, trace out ancilla,
        # then to feed into next step we must re-encode the (possibly mixed) density into a pure state by purification?
        # Simpler and consistent approach here: we simulate the global unitary on system+ancilla and then trace ancilla,
        # but to compose two steps we need to carry the full system+ancilla state through; that requires expanding ancilla spaces.
        # For brevity we re-apply each unitary starting from original psi and assume composition by applying all unitaries
        # on the same system+ancilla (i.e., effectively using same ancilla space and applying U_step1 U_step2 ...).
        # Implementation: build a composite unitary by combining param layers and run once.
        # Here we instead simulate sequential application by building a temporary device and applying all layers.
        dev_comp = qml.device(self.backend, wires=self.total_wires)
        @qml.qnode(dev_comp, interface="autograd")
        def composite_qnode(all_thetas_flat, state_vec):
            # prepare input
            qml.templates.MottonenStatePreparation(state_vec, wires=self.sys_wires)
            # apply all step HEAs in sequence
            offset = 0
            for step in range(self.timesteps):
                theta_shape = self.params[step].shape
                size = theta_shape[0] * theta_shape[1]
                theta_flat = all_thetas_flat[offset:offset+size]
                theta = theta_flat.reshape(theta_shape)
                self._apply_heas(theta)
                offset += size
            return qml.density_matrix(self.sys_wires)

        # gather all params flattened
        all_flat = np.concatenate([p.reshape(-1) for p in self.params])
        rho_pred = composite_qnode(all_flat, psi)
        return rho_pred

    def apply_total_unitary(self):
        """
        (Optional) Return a QNode that implements the composed unitary U_total = U_timestep * ... * U_1
        on (system + ancilla). This QNode can be used for diagnostics or to extract the full unitary
        (if device supports unitary extraction).
        """
        dev_comp = qml.device(self.backend, wires=self.total_wires)
        @qml.qnode(dev_comp, interface="autograd")
        def unitary_qnode(all_thetas_flat):
            # prepare computational basis state |0...0>, then apply layers to get full unitary action
            # However extracting full matrix requires device-specific capabilities; here we return state
            # from applying U to basis states externally if needed.
            # We'll just apply and return density on all wires for now.
            # prepare zero state implicitly
            offset = 0
            for step in range(self.timesteps):
                theta_shape = self.params[step].shape
                size = theta_shape[0] * theta_shape[1]
                theta_flat = all_thetas_flat[offset:offset+size]
                theta = theta_flat.reshape(theta_shape)
                self._apply_heas(theta)
                offset += size
            return qml.density_matrix(range(self.n_total))
        return unitary_qnode
# %%

#%%
if __name__ == "__main__":
    import pennylane as qml
    from pennylane import numpy as np
    from scipy.linalg import sqrtm

    # ---- 导入刚才的 QuantumFlowTomography 类 ----
    

    # 定义目标量子过程：Hadamard gate
    def hadamard_process(state_vec):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        return H @ state_vec

    # 生成训练数据
    def generate_dataset(num_samples=8):
        samples_input = []
        samples_output = []
        for _ in range(num_samples):
            # 随机 Bloch 球上的单比特态
            theta, phi = np.random.rand() * np.pi, np.random.rand() * 2 * np.pi
            psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
            psi_out = hadamard_process(psi)
            samples_input.append(psi)
            samples_output.append(psi_out)
        return np.array(samples_input), np.array(samples_output)

    # 创建数据
    samples_input, samples_output = generate_dataset(num_samples=100)

    # 初始化 QuantumFlowTomography
    qft = QuantumFlowTomography(
        n_qubits_in=1,
        n_qubits_out=1,
        depth=1,          # 每一步只用一个浅层 HEA
        timesteps=3,      # flow matching 分 3 步训练
        shots=0,
        backend="default.qubit"
    )

    # 训练
    qft.fit(samples_input, samples_output, epochs_per_step=500, lr=0.05, verbose=True)

    # 测试：选择一个输入态
    test_state = np.array([1.0, 0.0])   # |0>
    rho_true = np.outer(hadamard_process(test_state), np.conjugate(hadamard_process(test_state)))
    rho_pred = qft.predict_density(test_state)

    def fidelity(rho, sigma):
        # 保证输入是 numpy array (密度矩阵)
        rho = np.array(rho, dtype=complex)
        sigma = np.array(sigma, dtype=complex)

        # 计算 sqrt(rho)
        sqrt_rho = sqrtm(rho)

        # 计算 sqrt(sqrt(rho) * sigma * sqrt(rho))
        inner = sqrtm(sqrt_rho @ sigma @ sqrt_rho)

        # 取实部，避免数值误差导致出现极小的虚部
        return np.real((np.trace(inner))**2)

    print("\nTrue output density:\n", rho_true)
    print("\nPredicted output density:\n", rho_pred)
    print("\nFidelity between true and predicted:", fidelity(rho_true, rho_pred))

# %%