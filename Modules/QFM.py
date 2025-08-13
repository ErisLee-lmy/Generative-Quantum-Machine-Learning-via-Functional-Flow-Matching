# Quantum Flow Matching (QFM) Module
# This module implements the Quantum Flow Matching algorithm for quantum state preparation.

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
    def __init__(self, input_samples, target_samples, device_class="default.qubit", 
                 num_ancilla = 2, num_layers = 4,
                 t_max = 1.0, num_steps = 20, 
                 learning_rate=0.01 , num_epochs=1000,notice=False):
        
        super().__init__()
        dtype = torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64
        # 自动检测 GPU
        if notice:
            print("CUDA available:", torch.cuda.is_available())

        if device_class == "default.qubit":
            try:
                _ = qml.device("lightning.gpu", wires=num_qubits)
                device_class = "lightning.gpu"
                if self.notice:
                    print("[INFO] GPU detected, using lightning.gpu backend for acceleration.")
            except Exception:
                if self.notice:
                    print("[WARNING] lightning.gpu not available, falling back to default.qubit.")
        
        
        # Define the quantum and classical device
        self.qdevice = qml.device(device_class, wires=self.num_total_qubits)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
             
        # Store input and target samples
        self.input_samples = input_samples
        self.target_samples = target_samples
        
        
        # Store quantum circuit parameters
        self.num_qubits = input_samples.shape[1]  # Assuming input_samples is of shape (batch_size, num_qubits)
        self.num_ancilla = num_ancilla
        self.num_layers = num_layers
        
        self.num_total_qubits = self.num_qubits + self.num_ancilla * self.num_layers
        
        


        
        # Store training parameters
        self.learning_rate = learning_rate
        self.batch_size = input_samples.shape[0]  # Assuming input_samples is of shape (batch_size, num_qubits)
        self.num_epochs = num_epochs
        self.notice = notice
        
        # Initialize parameters for the quantum circuit
        self.params_shape = (self.num_layers, self.num_qubits, 3)
        init_params = (2 * torch.rand(self.params_shape) - 1) * torch.pi
        self.parametters = nn.Parameter(init_params, requires_grad=True)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        
        # Time evolution parameters
        self.t_max = t_max
        self.num_steps = num_steps
        self.t_values = torch.linspace(0, self.t_max, self.num_steps).to(self.device)
        self.t_step = self.t_max / self.num_steps
        
        
        dtype = torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64

        x_in = self.input_samples
        x_out = self.target_samples
        # convert numpy -> torch if needed
        if not isinstance(x_in, torch.Tensor):
            x_in = torch.tensor(x_in, dtype=torch.get_default_dtype(), device=self.device)
        if not isinstance(x_out, torch.Tensor):
            x_out = torch.tensor(x_out, dtype=torch.get_default_dtype(), device=self.device)

        # Ensure complex type for amplitudes
        if not torch.is_complex(x_in):
            x_in = x_in.to(self.device).to(dtype=torch.cfloat) if x_in.dtype.is_floating_point else x_in
        if not torch.is_complex(x_out):
            x_out = x_out.to(self.device).to(dtype=torch.cfloat) if x_out.dtype.is_floating_point else x_out

        batch_size = x_in.shape[0]
        # infer system dimension
        
        self.d_sys = 2 ** self.num_qubits

        # build empirical density matrices rho0, rho1 (torch)
        rho0 = torch.zeros((self.d_sys, self.d_sys), dtype=dtype, device=self.device)
        rho1 = torch.zeros((self.d_sys, self.d_sys), dtype=dtype, device=self.device)
        for i in range(batch_size):
            psi0 = x_in[i].reshape(self.d_sys, 1)
            psi1 = x_out[i].reshape(self.d_sys, 1)
            rho0 = rho0 + psi0 @ psi0.conj().t()
            rho1 = rho1 + psi1 @ psi1.conj().t()
        rho0 = rho0 / batch_size
        rho1 = rho1 / batch_size
        
        self.rho0 = rho0
        self.rho1 = rho1
        
    def PQC(self, input):
        '''
        Variational Parameterized Quantum Circuit (PQC) for QFM.
        Args:
            input: Input tensor of shape (batch_size, num_qubits)
        Returns:
            qml.state(): Quantum state after applying the circuit
            tuple: List of unitary matrices for each layer
        '''
        
        @qml.qnode(qml.device("default.qubit"), interface="torch")
        def circuit():
            # 存储每层的矩阵
            U_total_list = []

            qml.AmplitudeEmbedding(features=input,
                                    wires=range(self.num_qubits),
                                    normalize=True)

            for i in range(self.num_layers):
                # 构建该层的 qfunc
                layer_fn = self.layer_with_unitary(i)
                # 获取该层矩阵（非可微）
                U_mat = qml.matrix(layer_fn)()
                U_total_list.append(U_mat)
                # 在主电路中也应用该层
                layer_fn()

            return qml.state(), tuple(U_total_list)  # tuple便于QNode返回
        
        
        output_state, U_total_list = circuit()
        return output_state, U_total_list
        
    def layer_with_unitary(self,layer_index):
        """
        single layer of the quantum circuit with unitary matrix extraction.
        Args:
            layer_index: Index of the layer to be constructed
        Returns:
            function: A function that applies the layer operations
        """
        i = layer_index
        params = self.parametters
        def _layer():
            ancilla_start = self.num_qubits + layer_index * self.num_ancilla
            ancilla_end = ancilla_start + self.num_ancilla
            # system part
            for j in range(self.num_qubits):
                qml.RX(params[i, j, 0], wires=j)
                qml.RY(params[i, j, 1], wires=j)
                qml.RZ(params[i, j, 2], wires=j)
                qml.CNOT(wires=[j, (j + 1) % self.num_qubits])  # 避免越界
            # ancilla part
            if self.num_ancilla > 0:
                for j in range(ancilla_start, ancilla_end):
                    qml.RX(params[i, j, 0], wires=j)
                    qml.RY(params[i, j, 1], wires=j)
                    qml.RZ(params[i, j, 2], wires=j)
                    qml.CNOT(wires=[j, (j + 1) % (self.num_qubits + self.num_ancilla)])
                # entangle system with ancilla
                for m in range(self.num_qubits):
                    for n in range(ancilla_start, ancilla_end):
                        qml.CNOT(wires=[m, n])
        return _layer
    
    def flow(self, input):
        """
        Compute the list of generator superoperators L_list corresponding to U_total_list from PQC.
        Returns:
            list of np.ndarray: Each element is the Liouville representation of L for one layer.
        """
        # 调用 PQC 得到 U_total_list
        _, U_total_list = self.PQC(input)

        L_list = []
        dim_sys = 2 ** self.num_qubits
        dim_anc = 2 ** self.num_ancilla

        # 初始 ancilla 状态 |0><0|
        ancilla_state = np.zeros((dim_anc, 1))
        ancilla_state[0, 0] = 1
        rho_anc = ancilla_state @ ancilla_state.conj().T

        for U in U_total_list:
            # 从 U 构造 Kraus 算子
            K_list = []
            for m in range(dim_anc):
                # ancilla basis |m>
                basis_m = np.zeros((dim_anc, 1))
                basis_m[m, 0] = 1
                proj_m = basis_m @ basis_m.conj().T
                # 投影到 ancilla 子空间并 trace out ancilla
                # reshape: 系统-ancilla 结构
                U_block = np.kron(np.eye(dim_sys), proj_m) @ U
                # 应用初始 ancilla 态
                K_m = U_block @ np.kron(np.eye(dim_sys), ancilla_state)
                # reshape 成系统维度方阵
                K_m = K_m.reshape(dim_sys, dim_sys)
                K_list.append(K_m)

            # 构造 Liouville 表示的超算符 S
            S = np.zeros((dim_sys**2, dim_sys**2), dtype=complex)
            for K in K_list:
                S += np.kron(K, K.conj())

            # 取对数得到生成元 L
            L = (scipy.linalg.logm(S) / self.t_step) # this step need t_step to be short enough
            L_list.append(L)

        return L_list

    def QuantumSkinhorn(self, epsilon=1e-2, max_iter_sinkhorn=200, tol_sinkhorn=1e-6, mu_reg=1e-6):
        """
        Torch-friendly approximate Quantum Sinkhorn that returns a time series of
        approximate Liouville generators L(t) and the coupling matrices Gamma(t).
        No CVX/SDP involved — everything runs on torch (GPU-capable).

        Returns:
            L_list: list of torch.Tensor, each shape (d^2, d^2)
            Gamma_list: list of torch.Tensor, each shape (d^2, d^2)
            rho_ts: list of torch.Tensor, intermediate marginals (d,d)
        """


        # move samples to device and convert to complex dtype
        device = self.device if hasattr(self, "device") else torch.device("cpu")
        dtype = torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64

        d = self.d_sys  # Dimension of the quantum system
        rho0 = self.rho0
        rho1 = self.rho1

        # helper: hermitian symmetrize
        def herm(A):
            return 0.5 * (A + A.conj().transpose(-2, -1))

        # helper: matrix log / exp via eigen-decomposition (torch)
        def mat_log_torch(A, reg=1e-12):
            A = herm(A)
            # add tiny identity to ensure positive eigenvalues
            vals, vecs = torch.linalg.eigh(A + reg * torch.eye(A.shape[-1], dtype=A.dtype, device=device))
            # clamp vals away from zero for numeric stability
            vals = torch.clamp(vals, min=1e-15)
            lvals = torch.log(vals)
            return (vecs * lvals.unsqueeze(0)) @ vecs.conj().transpose(-2, -1)

        def mat_exp_torch(A):
            A = herm(A)
            vals, vecs = torch.linalg.eigh(A)
            evals = torch.exp(vals)
            return (vecs * evals.unsqueeze(0)) @ vecs.conj().transpose(-2, -1)

        # build SWAP operator on system⊗system (torch)
        # SWAP = sum_{i,j} |i,j><j,i|
        # caution: loops over d^2, okay for small d; for larger d you may want optimized construction
        eye_d = torch.eye(d, dtype=dtype, device=device)
        SWAP = torch.zeros((d*d, d*d), dtype=dtype, device=device)
        for i in range(d):
            for j in range(d):
                ei = torch.zeros(d, dtype=dtype, device=device); ei[i] = 1.0
                ej = torch.zeros(d, dtype=dtype, device=device); ej[j] = 1.0
                SWAP += torch.kron(ei.unsqueeze(1) @ ej.unsqueeze(0), ej.unsqueeze(1) @ ei.unsqueeze(0))
        C = torch.eye(d*d, dtype=dtype, device=device) - SWAP

        # kernel
        K = mat_exp_torch(-C / epsilon)   # (d^2, d^2) Hermitian PD

        # partial traces (both subsystems dim = d)
        def partial_trace_right(X):
            # X shape (d*d, d*d) interpret as (d, d, d, d) with indices (i,a ; j,b)
            X4 = X.reshape(d, d, d, d)
            # Tr_B -> sum_a X[i,a, j,a] -> shape (d,d)
            return torch.einsum('iaja->ij', X4)

        def partial_trace_left(X):
            X4 = X.reshape(d, d, d, d)
            # Tr_A -> sum_i X[i,a, i,b] -> shape (d,d)
            return torch.einsum('iaib->ab', X4)

        # time grid and simple interpolation of marginals
        N = int(self.num_steps)
        t_vals = torch.linspace(0.0, 1.0, N + 1, device=device)
        rho_ts = [ (1.0 - float(t.item())) * rho0 + float(t.item()) * rho1 for t in t_vals ]

        Gamma_list = []
        L_list = []

        # precompute logK if desired (we will use full matrix multiplies, simpler)
        # logK = mat_log_torch(K)

        # Sinkhorn iterations for each time interval
        for k in range(N):
            rho_a = rho_ts[k]
            rho_b = rho_ts[k+1]

            # initialize U, V as identity (d^2 x d^2)
            U = torch.eye(d*d, dtype=dtype, device=device)
            V = torch.eye(d*d, dtype=dtype, device=device)

            # logs of marginals (used in updates)
            log_rho_a = mat_log_torch(rho_a)
            log_rho_b = mat_log_torch(rho_b)

            for it in range(max_iter_sinkhorn):
                Gamma = U @ K @ V
                A = partial_trace_right(Gamma)
                B = partial_trace_left(Gamma)

                err = torch.norm(A - rho_a) + torch.norm(B - rho_b)
                if err.item() < tol_sinkhorn:
                    break

                # compute logA and logB
                logA = mat_log_torch(A)
                logB = mat_log_torch(B)

                # left update: U <- exp( (log_rho_a - logA) ⊗ I ) @ U
                Delta_left = torch.kron((log_rho_a - logA), torch.eye(d, dtype=dtype, device=device))
                U = mat_exp_torch(Delta_left) @ U

                # right update: V <- V @ exp( I ⊗ (log_rho_b - logB)^T )
                Delta_right = torch.kron(torch.eye(d, dtype=dtype, device=device),
                                        (log_rho_b - logB).T)
                V = V @ mat_exp_torch(Delta_right)

            # final Gamma_k
            Gamma_k = U @ K @ V
            # symmetrize and ensure Hermitian PSD via eigen cutoff
            Gamma_k = herm(Gamma_k)
            w, v = torch.linalg.eigh(Gamma_k)
            w_clamped = torch.clamp(w, min=0.0)
            Gamma_k = (v * w_clamped.unsqueeze(0)) @ v.conj().transpose(-2, -1)

            Gamma_list.append(Gamma_k)

            # --- build superoperator S_k from (approx) Choi = Gamma_k ---
            # For Choi-like J (shape d^2 x d^2) with ordering (out_idx, in_idx),
            # action on basis E_pq: M = Tr_A[ J (I ⊗ E_pq^T) ]
            S_k = torch.zeros((d*d, d*d), dtype=dtype, device=device)
            # build basis E_pq and fill columns
            for p in range(d):
                for q in range(d):
                    E_pq = torch.zeros((d, d), dtype=dtype, device=device)
                    E_pq[p, q] = 1.0 + 0j
                    RHS = torch.kron(torch.eye(d, dtype=dtype, device=device), E_pq.T)
                    tmp = Gamma_k @ RHS
                    M = partial_trace_right(tmp)
                    vec_M = M.reshape(d*d)
                    col_idx = p * d + q
                    S_k[:, col_idx] = vec_M

            # approximate generator L_k ~ (S_k - I)/Δt
            delta_t = self.t_step if hasattr(self, "t_step") else (1.0 / self.num_steps)
            Id = torch.eye(d*d, dtype=dtype, device=device)
            L_k = (S_k - Id) / delta_t
            L_list.append(L_k)

        return L_list, Gamma_list, rho_ts
    
    
    def compare_unitaries_choi(self, input, L_list, norm_type="fro"):
        """
        Compare PQC output unitaries with approximate Quantum Sinkhorn unitaries using Choi matrices.

        Args:
            input: input_samples, shape (batch_size, num_qubits)
            norm_type: "fro" (Frobenius), "trace" (trace norm)

        Returns:
            distances: list of floats, one per layer
        """
        _, U_total_list = self.PQC(input)
        

        # 系统维度
        dim_sys = 2 ** self.num_qubits
        dim_anc = 2 ** self.num_ancilla

        # PQC 与 Sinkhorn 的 Choi 矩阵列表
        choi_pqc_list = []
        choi_sink_list = []

        # Helper: 将 U_total (system+ancilla) 转为 Choi 矩阵
        def unitary_to_choi(U):
            # 投影初始 ancilla |0>
            ancilla_state = np.zeros((dim_anc,1))
            ancilla_state[0,0] = 1
            K_list = []
            for m in range(dim_anc):
                basis_m = np.zeros((dim_anc,1))
                basis_m[m,0] = 1
                proj_m = basis_m @ basis_m.conj().T
                # 系统-ancilla 结构
                U_block = np.kron(np.eye(dim_sys), proj_m) @ U
                K_m = U_block @ np.kron(np.eye(dim_sys), ancilla_state)
                K_m = K_m.reshape(dim_sys, dim_sys)
                K_list.append(K_m)
            # 构造 Choi: J = sum_k |K_k><K_k|
            J = np.zeros((dim_sys**2, dim_sys**2), dtype=complex)
            for K in K_list:
                J += np.kron(K, K.conj())
            return J

        for U_pqc, L in zip(U_total_list, L_list):
            # PQC Choi
            J_pqc = unitary_to_choi(U_pqc)
            choi_pqc_list.append(J_pqc)
            # Sinkhorn Choi
            U_sink = scipy.linalg.expm(L.cpu().numpy() * self.t_step)
            J_sink = unitary_to_choi(U_sink)
            choi_sink_list.append(J_sink)

        # 计算 Choi 距离
        distances = []
        for J1, J2 in zip(choi_pqc_list, choi_sink_list):
            diff = J1 - J2
            if norm_type == "fro":
                d = np.linalg.norm(diff, 'fro')
            elif norm_type == "trace":
                d = np.sum(np.linalg.svd(diff, compute_uv=False))
            else:
                raise ValueError("Unsupported norm_type")
            distances.append(d)

        return distances

    def forward(self, input_samples=None, norm_type="fro"):
        """
        Compute distance between PQC output and Quantum Sinkhorn output (Choi matrices).
        
        Args:
            input_samples: optional torch.Tensor, shape (batch_size, 2^n)
            norm_type: "fro" or "trace"
            
        Returns:
            torch.Tensor: total distance (scalar) or per-layer distances
        """
        # use default input_samples if not provided
        if input_samples is None:
            input_samples = self.input_samples
        input_samples = input_samples.to(self.device)

        # 1. PQC 输出态
        pqc_state, _ = self.PQC(input_samples)  # shape (d_sys, ) or (batch, d_sys)

        # 2. Sinkhorn 输出生成器 L_list
        L_list, _, _ = self.QuantumSkinhorn()   # L_list: list of (d^2, d^2) torch tensors

        # 3. PQC -> Choi
        # TODO: 这里建议 torch 版 unitary_to_choi，避免 np/scipy
        dim_sys = 2 ** self.num_qubits
        dim_anc = 2 ** self.num_ancilla

        choi_pqc_list = []
        choi_sink_list = []

        # 构造 ancilla 初态 |0>
        ancilla_state = torch.zeros((dim_anc, 1), dtype=pqc_state.dtype, device=self.device)
        ancilla_state[0, 0] = 1.0

        for U_mat, L in zip(_, L_list):  # PQC 的 unitary list
            # PQC Choi (torch 实现)
            # 可以直接用: J = sum_k kron(K_k, K_k.conj()), K_k 由 partial trace 生成
            # 这里省略具体实现，参考之前的 unitary_to_choi，但用 torch.kron
            J_pqc = self.unitary_to_choi_torch(U_mat, ancilla_state)
            choi_pqc_list.append(J_pqc)

            # Sinkhorn Choi
            U_sink = torch.matrix_exp(L * self.t_step)
            J_sink = self.unitary_to_choi_torch(U_sink, ancilla_state)
            choi_sink_list.append(J_sink)

        # 4. 计算距离
        distances = []
        for J1, J2 in zip(choi_pqc_list, choi_sink_list):
            diff = J1 - J2
            if norm_type == "fro":
                d = torch.linalg.norm(diff)
            elif norm_type == "trace":
                d = torch.sum(torch.linalg.svd(diff, full_matrices=False).S)
            distances.append(d)

        distances = torch.stack(distances)
        return distances.mean()  # scalar loss


        