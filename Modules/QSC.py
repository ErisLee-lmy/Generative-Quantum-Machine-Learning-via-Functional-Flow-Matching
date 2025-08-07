# %%
import pennylane as qml
from pennylane import numpy as qnp
import numpy as np   # 标准 numpy 用于 QR 分解
import matplotlib.pyplot as plt

sqrt2 = np.sqrt(2)

class QSC:
    def __init__(self, n_qubits: int, num_layers: int = 3,initial_state=None,
                 device_class="lightning.gpu",notice = True):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.initial_state = initial_state
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

        self.dev = qml.device(device_class, wires=n_qubits)

        # 处理初始态（支持纯态或密度矩阵）
        self.initial_state = self.process_initial_state()

        # 定义 QNode
        @qml.qnode(self.dev)
        def qsc_circuit():
            self.input()
            self.circuit()
            return qml.state()  # 始终输出纯态向量

        self.output = qsc_circuit

    def process_initial_state(self):
        """将输入处理为纯态向量"""
        dim = 2 ** self.n_qubits
        state = self.initial_state
        # 情况 1: 输入 None → 默认 |000...0>
        if state is None:
            vec = qnp.zeros(dim)
            vec[0] = 1.0
            return vec

        # 情况 2: 输入是向量 → 检查归一化
        state = np.array(state)
        if state.ndim == 1:
            # 纯态向量
            norm = np.linalg.norm(state)
            if not np.isclose(norm, 1):
                state = state / norm
            return qnp.array(state, dtype=complex)

        # 情况 3: 输入是密度矩阵
        if state.ndim == 2:
            # 检查是否纯态密度矩阵 (rank=1)
            eigvals, eigvecs = np.linalg.eigh(state)
            # 取最大特征值对应的特征向量
            max_idx = np.argmax(eigvals)
            pure_vec = eigvecs[:, max_idx]
            # 归一化
            pure_vec = pure_vec / np.linalg.norm(pure_vec)
            return qnp.array(pure_vec, dtype=complex)

        raise ValueError("Initial state must be a vector or density matrix.")

    def input(self):
        """加载纯态向量"""
        qml.StatePrep(self.initial_state, wires=range(self.n_qubits))

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

def frame_potential(qsc_class, num_samples=20, k=2):
    """估计k阶frame potential, 用于评估接近Haar程度"""
    states = []
    unitaries = []

    # 收集多个随机电路的unitary
    for _ in range(num_samples):
        # 输出最终态向量（单位化的列）
        psi = qsc_class.output()
        # 将其视为列向量嵌入unitary（只做比较用）
        unitaries.append(np.outer(psi, np.conj(psi)))

    # 计算 frame potential
    F = 0
    for i in range(num_samples):
        for j in range(num_samples):
            # Tr(U_i^† U_j)
            overlap = np.trace(np.dot(np.conjugate(unitaries[i].T), unitaries[j]))
            F += np.abs(overlap) ** (2 * k)

    F /= num_samples ** 2
    return F




if "__main__" == __name__:

    # 测试输入
    n_qubits = 1
    n_samples = 300
    initial_state = [1, 0]
    
    
    # 归一化初始态
    initial_state = [i / np.linalg.norm(initial_state) for i in initial_state]
    qsc = QSC(n_qubits=n_qubits, num_layers=3,device_class="default.qubit",initial_state=initial_state)
    initial_state = qsc.process_initial_state()
    
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
    

    
    x_initial = 2 * np.real(np.conj(initial_state[0]) * initial_state[1])
    y_initial = 2 * np.imag(np.conj(initial_state[0]) * initial_state[1])
    z_initial= np.abs(initial_state[0])**2 - np.abs(initial_state[1])**2
    ax.scatter([x_initial], [y_initial], [z_initial], color = 'red', s=80, label='Initial State')
    

    # 设置
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Bloch Sphere Representation for State Vector Input")
    plt.show()
    
    # 计算 frame potential
    num_layers_list = range(1,20)
    frame_potential_list = []
    for num in num_layers_list:
        qsc_sample = QSC(n_qubits=10,num_layers=num,notice=False)
        F = frame_potential(qsc, num_samples=100, k=2)
        frame_potential_list.append(F)

    plt.figure()
    plt.plot(num_layers_list,frame_potential_list,color = 'blue')
    plt.scatter(num_layers_list,frame_potential_list,marker='o',color='red')
    plt.title('number of layers VS frame potenrial for n_qubits = 10')
    plt.xlabel('number of layers')
    plt.ylabel('frame potential')
    plt.grid()
    plt.show()
    
# %%
