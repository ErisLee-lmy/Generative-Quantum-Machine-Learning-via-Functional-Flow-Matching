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
                 num_qubits = 4 ,num_ancilla = 2, num_layers = 4,
                 t_max = 1.0, num_steps = 20, 
                 learning_rate=0.01 , batch_size=64, num_epochs=1000,notice=False):
        
        super().__init__()
        
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
                    
        # Store input and target samples
        self.input_samples = input_samples
        self.target_samples = target_samples
        
        
        # Store quantum circuit parameters
        self.num_qubits = num_qubits
        self.num_ancilla = num_ancilla
        self.num_layers = num_layers
        
        self.num_total_qubits = self.num_qubits + self.num_ancilla * self.num_layers
        
        # Define the quantum device
        self.qdevice = qml.device(device_class, wires=self.num_total_qubits)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
            L = (scipy.linalg.logm(S) / self.t_step)
            L_list.append(L)

        return L_list

    
 
    