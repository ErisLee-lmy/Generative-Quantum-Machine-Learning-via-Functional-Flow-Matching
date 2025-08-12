# %%# Brickwork Quantum Circuit

import pennylane as qml
import numpy as np


class Brickwork:
    def __init__(self, n_qubits, depth, device_class="default.qubit", seed = None, notice=False):
        # 自动检测 GPU
        if device_class == "default.qubit":
            try:
                _ = qml.device("lightning.gpu", wires=n_qubits)
                device_class = "lightning.gpu"
                if notice:
                    print("[INFO] GPU detected, using lightning.gpu backend for acceleration.")
            except Exception:
                if notice:
                    print("[WARNING] lightning.gpu not available, falling back to default.qubit.")
                    
        self.device = qml.device(device_class, wires=n_qubits)
        self.n_qubits = n_qubits
        self.depth = depth
        self.notice = notice

        @qml.qnode(self.device)
        def circuit():
            for layer in range(depth):
                # 随机单比特旋转
                for w in range(n_qubits):
                    a, b, c = np.random.uniform(0, 2*np.pi, 3)
                    qml.RZ(a, wires=w)
                    qml.RY(b, wires=w)
                    qml.RZ(c, wires=w)
                # 交错 CNOT（brickwork）
                for w in range(0, n_qubits-1, 2):
                    qml.CNOT(wires=[w, w+1])
                for w in range(1, n_qubits-1, 2):
                    qml.CNOT(wires=[w, w+1])
            return qml.state()
        
        self.output = circuit()

if __name__ == "__main__":
    
# %%