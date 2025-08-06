# %%

import pennylane as qml
from pennylane import numpy as qnp
import numpy as np   # 标准 numpy 用于 QR 分解
import matplotlib.pyplot as plt

sqrt2 = np.sqrt(2)

class QSC:
    def __init__(self, n_qubits: int, num_layers: int = 20,
                 device_class="lightning.gpu", initial_state=None):
        """
        多层 Haar Unitary Scrambling Circuit

        Args:
            n_qubits: 比特数
            num_layers: Haar Unitary 层数
            device_class: 量子模拟器后端
            initial_state: 初始态（默认为 |000...0>）
        """
        self.n_qubits = n_qubits
        self.num_layers = num_layers

        # 自动检测 GPU
        if device_class == "default.qubit":
            try:
                _ = qml.device("lightning.gpu", wires=n_qubits)
                device_class = "lightning.gpu"
                print("[INFO] GPU detected, using lightning.gpu backend for acceleration.")
            except Exception:
                print("[WARNING] lightning.gpu not available, falling back to default.qubit.")

        self.dev = qml.device(device_class, wires=n_qubits)
        self.initial_state = initial_state
        
        # 定义 QNode
        @qml.qnode(self.dev)
        def qsc_circuit():
            self.input()
            self.circuit()  # 多层 Haar 随机 unitary
            return qml.state()

        self.output = qsc_circuit

    def input(self):
        """设置初始态"""
        if self.initial_state is None:
            initial_state = qnp.zeros(2 ** self.n_qubits)
            initial_state[0] = 1.0
        else:
            initial_state = self.initial_state
        qml.StatePrep(initial_state, wires=range(self.n_qubits))

    def haar_random_unitary(self):
        """生成 Haar 随机 unitary 矩阵"""
        dim = 2 ** self.n_qubits
        z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / sqrt2
        q, r = np.linalg.qr(z)
        d = np.diagonal(r)
        ph = d / np.abs(d)
        q = q * ph
        return q

    def circuit(self):
        """应用多层 Haar 随机 unitary"""
        for _ in range(self.num_layers):
            U = self.haar_random_unitary()
            qml.QubitUnitary(U, wires=range(self.n_qubits))



if "__main__" == __name__:

    # 测试输入
    n_qubits = 1
    n_samples = 400
    
    initial_state_list = [1, 0]
    qsc = QSC(n_qubits=n_qubits, device_class="default.qubit")
    
    
    initial_state_list = [i / np.linalg.norm(initial_state_list) for i in initial_state_list]
    initial_state = qnp.array(initial_state_list, dtype=complex)  # |0>
    
    states_list = []
    
    for i in range(n_samples):
        qsc.initial_state = initial_state
        state = qsc.output()
        states_list.append(state)
    # 1. 归一化（确保数值稳定）
    states_list = [state / np.linalg.norm(state) for state in states_list]
    states_list = np.array(states_list)

    # 2. 计算 Bloch 坐标
    alpha, beta = states_list[:,0], states_list[:,1]
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2

    # 3. 绘制 Bloch 球
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    # 球面网格
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(X, Y, Z, color='c', alpha=0.1, edgecolor='gray')

    # 坐标轴
    ax.quiver(0,0,0,1,0,0,color='black',arrow_length_ratio=0.1) # X轴
    ax.quiver(0,0,0,0,1,0,color='black',arrow_length_ratio=0.1) # Y轴
    ax.quiver(0,0,0,0,0,1,color='black',arrow_length_ratio=0.1) # Z轴

    # 绘制点
    ax.scatter([x], [y], [z], color='b', s=40)
    

    
    x_initial = 2 * np.real(np.conj(initial_state_list[0]) * initial_state_list[1])
    y_initial = 2 * np.imag(np.conj(initial_state_list[0]) * initial_state_list[1])
    z_initial= np.abs(initial_state_list[0])**2 - np.abs(initial_state_list[1])**2
    ax.scatter([x_initial], [y_initial], [z_initial], color = 'red', s=80, label='Initial State')
    

    # 设置
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Bloch Sphere Representation")
    plt.show()
# %%
