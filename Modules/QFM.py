
# Quantum Flow Matching (QFM) Module
# This module implements the Quantum Flow Matching algorithm for quantum state preparation.

# %%
import numpy as np
import scipy.linalg
import pennylane as qml
from pennylane import numpy as qnp
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time

class QFM(nn.Module):
    def __init__(self, input_samples = None, target_samples = None, device_class="default.qubit",
                 num_ancilla = 2, num_layers = 4,
                 t_max = 1.0, num_steps = 20,
                 learning_rate=0.01 , num_epochs=100, notice=False):

        super().__init__()

        # --- basic settings and device ---
        # infer system qubits from input samples shape
        self.num_qubits = int(np.log2(input_samples.shape[1]))
        self.num_ancilla = int(num_ancilla)
        self.num_layers = int(num_layers)
        # total wires (system + ancilla per layer)
        self.num_total_qubits = self.num_qubits + self.num_ancilla * self.num_layers

        self.notice = notice
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # complex dtype selection
        self.cfloat = torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64
        self.rdtype = torch.get_default_dtype()

        if self.notice:
            print("Torch device:", self.device, "complex dtype:", self.cfloat)

        # --- try to create PennyLane device (prefer lightning.gpu if available) ---
        chosen_qml_device = device_class
        qml_dev = None
        try:
            if device_class == "default.qubit":
                # Try lightning.gpu first (fast if installed)
                try:
                    qml_dev = qml.device("lightning.gpu", wires=self.num_total_qubits )
                    chosen_qml_device = "lightning.gpu"
                    if self.notice:
                        print("[INFO] Using pennylane lightning.gpu device.")
                except Exception:
                    # fallback to default.qubit
                    qml_dev = qml.device("default.qubit", wires=self.num_total_qubits )
                    chosen_qml_device = "default.qubit"
                    if self.notice:
                        print("[WARNING] lightning.gpu not available; using default.qubit.")
            else:
                qml_dev = qml.device(device_class, wires=self.num_total_qubits )
                chosen_qml_device = device_class
        except Exception as e:
            # ultimate fallback
            qml_dev = qml.device("default.qubit", wires=self.num_total_qubits )
            chosen_qml_device = "default.qubit"
            if self.notice:
                print("[ERROR] Could not create requested qml device, fallback to default.qubit:", e)

        self.qdevice = qml_dev
        self.qdevice_name = chosen_qml_device

        # --- store sample data as torch tensors on the chosen device ---
        self.input_samples = torch.as_tensor(input_samples, dtype=self.rdtype, device=self.device)
        self.target_samples = torch.as_tensor(target_samples, dtype=self.rdtype, device=self.device)

        # ensure complex amplitudes
        if not torch.is_complex(self.input_samples):
            self.input_samples = self.input_samples.to(self.device).to(self.cfloat)
        if not torch.is_complex(self.target_samples):
            self.target_samples = self.target_samples.to(self.device).to(self.cfloat)

        self.batch_size = int(self.input_samples.shape[0])

        # system dimension
        self.d_sys = int(self.input_samples.shape[1])


        # --- variational parameters ---
        # allocate parameters for system + ancilla wires
        self.params_shape = (self.num_layers, self.num_total_qubits, 3)
        init_params = (2 * torch.rand(self.params_shape, device=self.device) - 1) * torch.pi
        self.parametters = nn.Parameter(init_params, requires_grad=True)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # time evolution params
        self.t_max = float(t_max)
        self.num_steps = int(num_steps)
        self.t_values = torch.linspace(0, self.t_max, self.num_steps, device=self.device)
        self.t_step = float(self.t_max / max(1, self.num_steps))

        # training params
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    # ------------------------------
    # PQC and layer construction
    # ------------------------------
    def layer_with_unitary(self, layer_index):
        """
        Return a function that applies the layer gates (to be used in QNode).
        Uses self.parametters indexed by full wire index.
        """
        i = int(layer_index)
        params = self.parametters

        def _layer():
            ancilla_start = self.num_qubits + layer_index * self.num_ancilla
            ancilla_end = ancilla_start + self.num_ancilla

            # system rotations + entangling
            for j in range(self.num_qubits):
                qml.RX(params[i, j, 0], wires=j)
                qml.RY(params[i, j, 1], wires=j)
                qml.RZ(params[i, j, 2], wires=j)
            # simple nearest-neighbor entangling on system
            for j in range(self.num_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
            if self.num_qubits > 1:
                # close the ring
                qml.CNOT(wires=[self.num_qubits - 1, 0])

            # ancilla part if any
            if self.num_ancilla > 0:
                for w in range(ancilla_start, ancilla_end):
                    qml.RX(params[i, w, 0], wires=w)
                    qml.RY(params[i, w, 1], wires=w)
                    qml.RZ(params[i, w, 2], wires=w)
                # simple entangling for ancilla
                for idx in range(ancilla_start, ancilla_end - 1):
                    qml.CNOT(wires=[idx, idx + 1])
                # entangle system with ancilla qubits
                for m in range(self.num_qubits):
                    for n in range(ancilla_start, ancilla_end):
                        qml.CNOT(wires=[m, n])

        return _layer

    def PQC(self, input_state):
        # First, compute per-layer unitaries (numpy) by calling qml.matrix on each layer function
        U_list_np = []
        for i in range(self.num_layers):
            layer_fn = self.layer_with_unitary(i)
            try:
                U_mat_np = qml.matrix(layer_fn)()
            except Exception:
                U_mat_np = np.eye(2 ** (self.num_total_qubits), dtype=complex)
            U_list_np.append(U_mat_np)
        # convert to torch
        U_list_torch = [torch.as_tensor(U, dtype=self.cfloat, device=self.device) for U in U_list_np]

        @qml.qnode(self.qdevice, interface='torch')
        def circuit(psi):
            qml.AmplitudeEmbedding(features=psi, wires=range(self.num_qubits), normalize=True)
            for i in range(self.num_layers):
                layer_fn = self.layer_with_unitary(i)
                layer_fn()
            return qml.state()

        # run circuit (batched handling as before)
        if input_state.ndim == 1 or input_state.shape[0] == 1:
            psi = input_state.reshape(self.d_sys)
            state_out = circuit(psi)
            state_out = torch.as_tensor(state_out, dtype=self.cfloat, device=self.device)
            return state_out, U_list_torch
        else:
            states = []
            for i in range(input_state.shape[0]):
                psi = input_state[i].reshape(self.d_sys)
                s = circuit(psi)
                s = torch.as_tensor(s, dtype=self.cfloat, device=self.device)
                states.append(s)
            states = torch.stack(states)
            # return states and the list of per-layer unitaries (from first sample — they're parameter-only so same for all samples)
            return states, U_list_torch

    # ------------------------------
    # flow() - original (numpy/scipy) path (kept for compatibility)
    # ------------------------------
    def flow(self, input):
        """
        Compute generator superoperators L_list from PQC unitaries.
        This function follows the original structure using numpy/scipy.
        Returns list of numpy arrays (each S/L in numpy).
        """
        # call PQC to get unitaries (we will convert to numpy here)
        _, U_total_list = self.PQC(input)  # U_total_list are torch tensors on device
        # move unitaries to cpu numpy
        U_list_np = [U.cpu().detach().numpy() if isinstance(U, torch.Tensor) else np.array(U) for U in U_total_list]

        L_list = []
        dim_sys = 2 ** self.num_qubits
        dim_anc = 2 ** self.num_ancilla

        # initial ancilla state |0><0|
        ancilla_state = np.zeros((dim_anc, 1), dtype=complex)
        ancilla_state[0, 0] = 1.0
        rho_anc = ancilla_state @ ancilla_state.conj().T

        for U in U_list_np:
            # Build Kraus operators from U (system-ancilla)
            K_list = []
            # reshape U into (s_out, a_out, s_in, a_in)
            U4 = U.reshape(dim_sys, dim_anc, dim_sys, dim_anc)
            for m in range(dim_anc):
                K_m = U4[:, m, :, 0]   # pick a_out = m, a_in = 0
                # ensure shape is (dim_sys, dim_sys)
                K_list.append(K_m)
            # Liouville superoperator S = sum_k kron(K, K^*)
            S = np.zeros((dim_sys**2, dim_sys**2), dtype=complex)
            for K in K_list:
                S += np.kron(K, K.conj())
            L = (scipy.linalg.logm(S) / self.t_step)
            L_list.append(L)

        return L_list

    # ------------------------------
    # Torch-based Quantum Sinkhorn (GPU-capable)
    # ------------------------------
    def QuantumSkinhorn(self, epsilon=1e-2, max_iter_sinkhorn=200, tol_sinkhorn=1e-6, mu_reg=1e-6):
        """
        Torch-friendly batched Quantum Sinkhorn.
        All time intervals are processed in parallel on GPU.

        Returns:
            L_list: list of torch.Tensor, shape (N, d^2, d^2)
            Gamma_list: list of torch.Tensor, shape (N, d^2, d^2)
            rho_ts: list of torch.Tensor, shape (N+1, d, d)
        """

        device = self.device if hasattr(self, "device") else torch.device("cpu")
        dtype = self.cfloat

        # --- prepare states ---
        x_in = self.input_samples
        x_out = self.target_samples
        if not isinstance(x_in, torch.Tensor):
            x_in = torch.tensor(x_in, dtype=torch.get_default_dtype(), device=device)
        if not isinstance(x_out, torch.Tensor):
            x_out = torch.tensor(x_out, dtype=torch.get_default_dtype(), device=device)
        if not torch.is_complex(x_in):
            x_in = x_in.to(device).to(dtype=torch.cfloat)
        if not torch.is_complex(x_out):
            x_out = x_out.to(device).to(dtype=torch.cfloat)

        batch_size, d = x_in.shape
        d = int(d)

        # empirical rho0, rho1
        rho0 = torch.zeros((d, d), dtype=dtype, device=device)
        rho1 = torch.zeros((d, d), dtype=dtype, device=device)
        for i in range(batch_size):
            psi0 = x_in[i].reshape(d, 1)
            psi1 = x_out[i].reshape(d, 1)
            rho0 += psi0 @ psi0.conj().t()
            rho1 += psi1 @ psi1.conj().t()
        rho0 = rho0 / batch_size
        rho1 = rho1 / batch_size

        # helper functions (batched)
        def herm(A):
            return 0.5 * (A + A.conj().transpose(-2, -1))

        def mat_log_torch(A, reg=1e-12):
            A = herm(A)
            vals, vecs = torch.linalg.eigh(A + reg * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device))
            vals = torch.clamp(vals, min=1e-15)
            lvals = torch.log(vals)
            return (vecs * lvals.unsqueeze(-2)) @ vecs.conj().transpose(-2, -1)

        def mat_exp_torch(A):
            A = herm(A)
            vals, vecs = torch.linalg.eigh(A)
            evals = torch.exp(vals)
            return (vecs * evals.unsqueeze(-2)) @ vecs.conj().transpose(-2, -1)

        def partial_trace_right(X):
            # X: (B, d^2, d^2)
            X4 = X.reshape(X.shape[0], d, d, d, d)
            return torch.einsum('biajb->bij', X4)

        def partial_trace_left(X):
            X4 = X.reshape(X.shape[0], d, d, d, d)
            return torch.einsum('biaib->bab', X4)

        # --- build cost kernel ---
        SWAP = torch.zeros((d*d, d*d), dtype=dtype, device=device)
        for i in range(d):
            for j in range(d):
                ei = torch.zeros(d, dtype=dtype, device=device); ei[i] = 1
                ej = torch.zeros(d, dtype=dtype, device=device); ej[j] = 1
                SWAP += torch.kron(ei[:, None] @ ej[None, :], ej[:, None] @ ei[None, :])
        C = torch.eye(d*d, dtype=dtype, device=device) - SWAP
        K = mat_exp_torch(-C / epsilon)  # (d^2, d^2)

        # --- time grid ---
        N = int(self.num_steps)
        t_vals = torch.linspace(0.0, self.t_max, N + 1, device=device)
        rho_ts = (1.0 - t_vals[:, None, None]/self.t_max) * rho0 + t_vals[:, None, None]/self.t_max * rho1  # (N+1, d, d)

        # rho_a, rho_b for all intervals
        rho_a = rho_ts[:-1]  # (N, d, d)
        rho_b = rho_ts[1:]   # (N, d, d)

        # --- initialize U, V for all intervals ---
        eye_dd = torch.eye(d*d, dtype=dtype, device=device)
        U = eye_dd.unsqueeze(0).repeat(N, 1, 1)  # (N, d^2, d^2)
        V = eye_dd.unsqueeze(0).repeat(N, 1, 1)

        log_rho_a = mat_log_torch(rho_a)
        log_rho_b = mat_log_torch(rho_b)

        # --- Sinkhorn iterations ---
        for it in range(max_iter_sinkhorn):
            Gamma = U @ K.unsqueeze(0) @ V  # (N, d^2, d^2)
            A = partial_trace_right(Gamma)
            B = partial_trace_left(Gamma)

            err = torch.norm(A - rho_a) + torch.norm(B - rho_b)
            if err.item() < tol_sinkhorn:
                break

            logA = mat_log_torch(A)
            logB = mat_log_torch(B)

            Delta_left = torch.kron(log_rho_a - logA, torch.eye(d, dtype=dtype, device=device))
            Delta_left = Delta_left.reshape(N, d*d, d*d)
            U = mat_exp_torch(Delta_left) @ U

            Delta_right = torch.kron(torch.eye(d, dtype=dtype, device=device), (log_rho_b - logB).transpose(-2, -1))
            Delta_right = Delta_right.reshape(N, d*d, d*d)
            V = V @ mat_exp_torch(Delta_right)

        # --- finalize Gamma ---
        Gamma = U @ K.unsqueeze(0) @ V
        Gamma = herm(Gamma)
        w, v = torch.linalg.eigh(Gamma)
        w = torch.clamp(w, min=0.0)
        Gamma = (v * w.unsqueeze(-2)) @ v.conj().transpose(-2, -1)

        # --- build S_k in batch ---
        S_k = torch.zeros((N, d*d, d*d), dtype=dtype, device=device)
        for p in range(d):
            for q in range(d):
                E_pq = torch.zeros((d, d), dtype=dtype, device=device)
                E_pq[p, q] = 1
                RHS = torch.kron(torch.eye(d, dtype=dtype, device=device), E_pq.T)
                tmp = Gamma @ RHS.unsqueeze(0)
                M = partial_trace_right(tmp)
                vec_M = M.reshape(N, d*d)
                col_idx = p * d + q
                S_k[:, :, col_idx] = vec_M

        delta_t = self.t_step if hasattr(self, "t_step") else (1.0 / self.num_steps)
        Id = torch.eye(d*d, dtype=dtype, device=device)
        L_list = (S_k - Id.unsqueeze(0)) / delta_t

        # 转成 list 形式保持原接口习惯
        return list(L_list), list(Gamma), list(rho_ts)


    # ------------------------------
    # helper: unitary -> choi (torch)
    # ------------------------------
    def unitary_to_choi_torch(self, U):
        device = self.device
        dtype = self.cfloat
        dim_sys = 2 ** self.num_qubits
        dim_anc = 2 ** self.num_ancilla

        # reshape U: (s_out, a_out, s_in, a_in)
        U4 = U.reshape(dim_sys, dim_anc, dim_sys, dim_anc)
        J = torch.zeros((dim_sys * dim_sys, dim_sys * dim_sys), dtype=dtype, device=device)
        for m in range(dim_anc):
            K_m = U4[:, m, :, 0]  # (d_sys, d_sys)
            J = J + torch.kron(K_m, K_m.conj())
        return J


    # ------------------------------
    # Compare unitaries via choi matrices
    # ------------------------------
    def compare_unitaries_choi(self, input, L_list, norm_type="fro"):
        """
        Compare PQC output unitaries with the ones implied by L_list using Choi matrices.
        L_list: list of superoperators (torch tensors) shape (d^2,d^2)
        Returns distances list (python floats)
        """
        # get PQC unitaries
        _, U_total_list = self.PQC(input)
        # ensure U_total_list are torch tensors on device
        U_total_list = [torch.as_tensor(U, dtype=self.cfloat, device=self.device) for U in U_total_list]

        # compute choi for PQC and for Sinkhorn-derived unitaries (via matrix exponential)
        distances = []
        for U_pqc, L in zip(U_total_list, L_list):
            J_pqc = self.unitary_to_choi_torch(U_pqc)
            # from generator L get U_sink = exp(L * dt) (superoperator -> convert to unitary in full space?)
            # Here we follow the original approach: exponentiate L (superoperator) to get S and then try to convert to a unitary U_sink in the enlarged Hilbert space.
            # A practical proxy: interpret exp(L*dt) as a superoperator S and attempt to get corresponding Choi J_sink by reshaping S (but to stay consistent with previous design, we exponentiate L and try to treat it as operation)
            S_sink = torch.matrix_exp(L * float(self.t_step))
            # We need to turn S_sink (superoperator) back to a Choi-like matrix J_sink.
            # Here a simple approach: treat S_sink as action on vectorized matrices and build J_sink by applying S_sink to basis E_pq (same as in QuantumSkinhorn)
            d = self.d_sys
            J_sink = torch.zeros((d * d, d * d), dtype=self.cfloat, device=self.device)
            for p in range(d):
                for q in range(d):
                    E_pq = torch.zeros((d, d), dtype=self.cfloat, device=self.device)
                    E_pq[p, q] = 1.0 + 0j
                    RHS = torch.kron(torch.eye(d, dtype=self.cfloat, device=self.device), E_pq.T)
                    tmp = S_sink @ RHS
                    # partial trace over right => recover action result M
                    M = tmp.reshape(d, d, d, d)
                    M = torch.einsum('iaja->ij', M)
                    vec_M = M.reshape(d * d)
                    col_idx = p * d + q
                    J_sink[:, col_idx] = vec_M
            # compute norm of difference
            diff = J_pqc - J_sink
            if norm_type == "fro":
                dval = torch.linalg.norm(diff).item()
            elif norm_type == "trace":
                s = torch.linalg.svd(diff, full_matrices=False).S
                dval = float(torch.sum(s).item())
            else:
                raise ValueError("Unsupported norm_type")
            distances.append(dval)
        return distances

    # ------------------------------
    # forward: produce loss = mean Choi distance
    # ------------------------------
    def forward(self, input_samples=None, norm_type="fro"):
        """
        Compute mean distance between PQC Choi matrices and QuantumSinkhorn Choi matrices.
        Returns scalar torch tensor (loss).
        """
        if input_samples is None:
            input_samples = self.input_samples
        input_samples = input_samples.to(self.device)

        # 1) PQC: get state(s) and unitary list
        pqc_state, U_total_list = self.PQC(input_samples)

        # 2) Quantum Sinkhorn -> L_list (torch)
        L_list, Gamma_list, rho_ts = self.QuantumSkinhorn()

        # 3) convert PQC unitaries & compare via choi (torch)
        # ensure U_total_list are torch tensors
        U_total_list = [torch.as_tensor(U, dtype=self.cfloat, device=self.device) for U in U_total_list]

        # compute choi for both and distance per layer
        dim_sys = self.d_sys
        distances = []
        for U_mat, L in zip(U_total_list, L_list):
            J_pqc = self.unitary_to_choi_torch(U_mat)
            # S_sink from L
            S_sink = torch.matrix_exp(L * float(self.t_step))
            # build J_sink similarly as done in QuantumSkinhorn
            d = dim_sys
            J_sink = torch.zeros((d * d, d * d), dtype=self.cfloat, device=self.device)
            for p in range(d):
                for q in range(d):
                    E_pq = torch.zeros((d, d), dtype=self.cfloat, device=self.device)
                    E_pq[p, q] = 1.0 + 0j
                    RHS = torch.kron(torch.eye(d, dtype=self.cfloat, device=self.device), E_pq.T)
                    tmp = S_sink @ RHS
                    M = tmp.reshape(d, d, d, d)
                    M = torch.einsum('iaja->ij', M)
                    vec_M = M.reshape(d * d)
                    col_idx = p * d + q
                    J_sink[:, col_idx] = vec_M
            diff = J_pqc - J_sink
            if norm_type == "fro":
                dval = torch.linalg.norm(diff)
            elif norm_type == "trace":
                s = torch.linalg.svd(diff, full_matrices=False).S
                dval = torch.sum(s)
            else:
                raise ValueError("Unsupported norm_type")
            distances.append(dval)
        distances = torch.stack(distances)
        return distances.mean()

"""
qfm_test_script.py

Test script for the provided QFM class (Quantum Flow Matching).

Instructions:
- Put this file in the same folder as your QFM class definition (or paste the QFM class above this script in the same file).
- Run with Python. PennyLane + chosen device must be installed and working.

What the script does:
- Builds a small test problem (num_qubits configurable).
- Uses a *simple* initial distribution (product single-qubit states like |0>, |+>, |i>)
  and a *complex* target distribution (Haar-random entangled states).
- Instantiates QFM with conservative small parameters (few ancilla / layers) so it
  runs reasonably on CPU for tests.
- Runs a short training loop, prints loss, and reports evaluation metrics:
  - Choi-distance per layer (compare_unitaries_choi)
  - Average state fidelity between PQC reduced system states and target samples
  - Average trace distance between PQC reduced system states and target samples
- Saves a checkpoint of the trained variational parameters.

Note: this script is intended as a compact, runnable *test harness* — tweak
num_layers, num_ancilla, batch_size, and num_epochs to explore capabilities.
"""


# ------------------------- helpers ---------------------------------

def random_haar_states(batch, d, dtype=np.complex128, seed=None):
    if seed is not None:
        np.random.seed(seed)
    z = (np.random.randn(batch, d) + 1j * np.random.randn(batch, d)).astype(dtype)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z /= norms
    return z


def product_single_qubit_pool():
    # Common, simple single-qubit states to build product states from
    return [
        np.array([1.0, 0.0], dtype=np.complex128),  # |0>
        np.array([0.0, 1.0], dtype=np.complex128),  # |1>
        1.0 / np.sqrt(2.0) * np.array([1.0, 1.0], dtype=np.complex128),  # |+>
        1.0 / np.sqrt(2.0) * np.array([1.0, -1.0], dtype=np.complex128),  # |->
        1.0 / np.sqrt(2.0) * np.array([1.0, 1j], dtype=np.complex128),  # |+i>
        1.0 / np.sqrt(2.0) * np.array([1.0, -1j], dtype=np.complex128),  # |-i>
    ]


def random_product_states(batch, num_qubits, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pool = product_single_qubit_pool()
    samples = []
    for _ in range(batch):
        factors = [pool[np.random.randint(len(pool))] for _ in range(num_qubits)]
        vec = factors[0]
        for f in factors[1:]:
            vec = np.kron(vec, f)
        samples.append(vec)
    return np.stack(samples)


# fidelity between pure state |psi> (vector) and density matrix rho: <psi|rho|psi>
# or general Uhlmann fidelity for two density matrices (torch)

def fidelity_pure_vs_rho(psi_vec, rho):
    # psi_vec: complex numpy or torch vector (d,) ; rho: (d,d) torch tensor or numpy
    if isinstance(rho, torch.Tensor):
        device = rho.device
        psi = torch.as_tensor(psi_vec, dtype=rho.dtype, device=device).reshape(-1, 1)
        val = (psi.conj().transpose(-2, -1) @ rho @ psi).real.item()
        return float(np.clip(val, 0.0, 1.0))
    else:
        psi = psi_vec.reshape(-1, 1)
        val = (psi.conj().T @ rho @ psi).item()
        return float(np.clip(val.real, 0.0, 1.0))


def trace_distance(rho1, rho2):
    # 0.5 * sum |eigvals(rho1-rho2)|; works for numpy or torch (returns python float)
    if isinstance(rho1, torch.Tensor):
        diff = rho1 - rho2
        vals = torch.linalg.eigvalsh(diff)
        return 0.5 * float(torch.sum(torch.abs(vals)).item())
    else:
        diff = rho1 - rho2
        vals = np.linalg.eigvalsh(diff)
        return 0.5 * float(np.sum(np.abs(vals)))


def reduce_system_density_from_full_state(psi_full, num_qubits_system, num_total_qubits):
    # psi_full can be numpy vector (d_total,) or torch tensor; returns rho_sys (torch tensor if input torch)
    d_sys = 2 ** num_qubits_system
    d_total = 2 ** num_total_qubits
    psi = psi_full
    using_torch = isinstance(psi_full, torch.Tensor)
    if using_torch:
        psi = psi_full.contiguous().view(d_total)
        psi_mat = psi.reshape(d_sys, -1)  # (d_sys, d_anc)
        rho_sys = psi_mat @ psi_mat.conj().transpose(-2, -1)
        return rho_sys
    else:
        psi = np.asarray(psi_full).reshape(d_total)
        psi_mat = psi.reshape(d_sys, -1)
        rho_sys = psi_mat @ psi_mat.conj().T
        return rho_sys


# ------------------------ main test harness --------------------------

def main():


if __name__ == '__main__':
    main()

# %%