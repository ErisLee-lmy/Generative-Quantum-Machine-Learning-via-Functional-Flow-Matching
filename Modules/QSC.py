# %%
import pennylane as qml
from pennylane import numpy as qnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from scipy.stats import beta as Beta
import time
import math

sqrt2 = np.sqrt(2)
class QSC:
    def __init__(self, n_qubits: int, num_layers: int = 3,
                 device_class="lightning.gpu", initial_state=None, 
                 notice=True, global_haar=True):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.initial_state = initial_state
        self.global_haar = global_haar
        self.notice = notice

        # 自动检测 GPU
        if device_class == "default.qubit":
            try:
                _ = qml.device("lightning.gpu", wires=n_qubits)
                device_class = "lightning.gpu"
                if self.notice:
                    print("[INFO] GPU detected, using lightning.gpu backend for acceleration.")
            except Exception:
                if self.notice:
                    print("[WARNING] lightning.gpu not available, falling back to default.qubit.")

        self.dev = qml.device(device_class, wires=n_qubits)

        # 处理初始态
        self.initial_state = self.process_initial_state()

        # 定义 QNode
        @qml.qnode(self.dev)
        def qsc_circuit():
            self.input()
            self.circuit()   # 多层局部随机层
            return qml.state()  # 始终输出纯态向量

        self.output = qsc_circuit

    def process_initial_state(self):
        """将输入处理为纯态向量"""
        dim = 2 ** self.n_qubits
        state = self.initial_state
        if state is None:
            vec = qnp.zeros(dim)
            vec[0] = 1.0
            return vec

        state = np.array(state)
        if state.ndim == 1:
            norm = np.linalg.norm(state)
            if not np.isclose(norm, 1):
                state = state / norm
            return qnp.array(state, dtype=complex)

        if state.ndim == 2:
            eigvals, eigvecs = np.linalg.eigh(state)
            max_idx = np.argmax(eigvals)
            pure_vec = eigvecs[:, max_idx]
            pure_vec = pure_vec / np.linalg.norm(pure_vec)
            return qnp.array(pure_vec, dtype=complex)

        raise ValueError("Initial state must be a vector or density matrix.")

    def input(self):
        """加载纯态向量"""
        qml.StatePrep(self.initial_state, wires=range(self.n_qubits))

    # 生成单比特 Haar 随机 SU(2) 门
    def random_single_qubit_unitary(self):
        """使用Euler角生成单比特Haar随机门"""
        # Haar measure: betax~ arccos(1-2u), alpha,gamma ~ uniform(0,2π)
        alpha = 2 * np.pi * np.random.rand()
        beta = np.arccos(1 - 2*np.random.rand())  # [0, π]
        gamma = 2 * np.pi * np.random.rand()
        return alpha, beta, gamma

    def haar_random_unitary(self):
        """生成 Haar 随机 unitary 矩阵"""
        if self.global_haar:
            dim = 2**self.n_qubits
        else:
            dim = 4
        z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / sqrt2
        q, r = np.linalg.qr(z)
        d = np.diagonal(r)
        ph = d / np.abs(d)
        q = q * ph
        return q
    

        
    def circuit(self):
        """
        若 global_haar 为 True，则直接使用全局 Haar unitary；
        否则使用局部双比特 Haar 随机门，交错排列，堆叠 num_layers 层。
        """
        if self.global_haar:
            U = self.haar_random_unitary()
            qml.QubitUnitary(U, wires=range(self.n_qubits))
        else:
            for layer in range(self.num_layers):
                # 交错层：偶数层作用于 (0,1), (2,3), ...
                #        奇数层作用于 (1,2), (3,4), ...
                start = 0 if layer % 2 == 0 else 1
                for i in range(start, self.n_qubits - 1, 2):
                    U = self.haar_random_unitary()  # 2 qubits => 4-dim Hilbert space
                    qml.QubitUnitary(U, wires=[i, i + 1])


def haar_random_state(d):
    """
    返回长度 d 的复列向量 psi，满足 ||psi||=1。该方法：复高斯向量归一化，
    等价于从 U(d) 的 Haar measure 抽取列向量。
    """
    z = (np.random.randn(d) + 1j * np.random.randn(d)) / np.sqrt(2)
    z = z.astype(np.complex128)
    psi = z / np.linalg.norm(z)
    return psi  # 1D complex numpy array

def frame_potential(qsc_class, num_samples=400, k=2):
    """估计第 k 阶 frame potential, 用于评估输出态是否接近 Haar 分布"""
    states = []

    for _ in range(num_samples):
        psi = qsc_class.output()
        states.append(psi)

    F = 0
    for i in range(num_samples):
        for j in range(num_samples):
            # 计算 |<ψ_i|ψ_j>|^{2k}
            overlap = np.vdot(states[i], states[j])  # <ψ_i|ψ_j>
            F += np.abs(overlap) ** (2 * k)

    F /= num_samples ** 2
    return F

def ks_test_haar(qsc_class, num_samples=200):
    """
    KS 检验：比较输出态的概率分布与 Haar 理论分布（Beta(1, d-1)）的一致性
    返回 KS 统计量 和 p 值
    """
    d = 2 ** qsc_class.n_qubits
    probs_all = []

    for _ in range(num_samples):
        psi = qsc_class.output()
        psi = psi / np.linalg.norm(psi)  # 归一化
        probs = np.abs(psi) ** 2
        probs_all.extend(probs)  # 收集所有幅度的模平方

    # Haar 边际分布是 Beta(1, d-1)
    ks_stat, p_val = kstest(probs_all, Beta(a=1, b=d-1).cdf)
    return ks_stat, p_val


if "__main__" == __name__:

    # 测试输入
    n_qubits = 1
    n_samples = 300
    initial_state = [1, 0]
    
    
    # 归一化初始态
    initial_state = [i / np.linalg.norm(initial_state) for i in initial_state]
    qsc = QSC(n_qubits=n_qubits, num_layers=3,device_class="default.qubit",initial_state=initial_state,global_haar=True)
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
    
    # KS 测试 Local vs Global Haar
    n_qubits = 6
    num_layers_list = range(2, 30, 1)
    ks_stat_list = []
    p_val_list = []

    for num in num_layers_list:
        time_start = time.time()
        qsc_sample = QSC(n_qubits=n_qubits, num_layers=num, notice=False, global_haar=False)
        ks_stat, p_val = ks_test_haar(qsc_sample, num_samples=200)
        ks_stat_list.append(ks_stat)
        p_val_list.append(p_val)
        print(f"[Local] layers={num}, KS stat={ks_stat:.4f}, p={p_val:.4e}, time={time.time()-time_start:.2f}s")

    # Global Haar
    time_start = time.time()
    qsc_global = QSC(n_qubits=n_qubits, num_layers=1, notice=False, global_haar=True)
    ks_stat_global, p_val_global = ks_test_haar(qsc_global, num_samples=200)
    print(f"[Global] KS stat={ks_stat_global:.4f}, p={p_val_global:.4e}, time={time.time()-time_start:.2f}s")

    # 理论参考
    print(f"Ideal Haar p-value 理论上应接近 1.0 (表示无法拒绝分布一致假设)")

    # 绘图
    plt.figure()
    plt.plot(num_layers_list, ks_stat_list, color='blue', label='Local KS statistic')
    plt.scatter(num_layers_list, ks_stat_list, marker='o', color='red')
    plt.axhline(y=ks_stat_global, color='green', linestyle='--', label=f'Global KS stat = {ks_stat_global:.3f}')
    plt.title(f"KS Statistic vs Number of Local Layers (n_qubits={n_qubits})")
    plt.xlabel('Number of Local Layers')
    plt.ylabel('KS Statistic (lower is better)')
    plt.grid()
    plt.legend()
    plt.show()
    
    # 计算 frame potential
    # O(t^10 n^2) is enough t = 2, n = n_qubits 
    frame_potential_list = []

    k = 2
    for num in num_layers_list:
        time_start = time.time()
        qsc_sample = QSC(n_qubits=n_qubits,num_layers=num,notice=False,global_haar=False)
        F = frame_potential(qsc_sample, num_samples=100, k=k)
        frame_potential_list.append(F)
        print(f"constructed QSC with {num} layers, frame potential: {F:.4e}, time: {time.time() - time_start:.2f}s")
    
    time_start = time.time()
    qsc_global = QSC(n_qubits=n_qubits,num_layers=num,notice=False,global_haar=True)
    F_global = frame_potential(qsc_global, num_samples=100, k=2)
    
    
    
    print(f"constructed QSC with global unitary transform, frame potential: {F_global:.4e}, time: {time.time() - time_start:.2f}s")
    
    F_2_haar = math.factorial(k) / ((2 ** n_qubits) * (2 ** n_qubits + 1))  # For k=2
    plt.figure()
    plt.plot(num_layers_list,frame_potential_list,color = 'blue',label='local layers')
    plt.scatter(num_layers_list,frame_potential_list,marker='o',color='red')
    plt.axhline(y=F_global, color='r', linestyle='--',label = f'global Haar with F = {F_global:.3e}')
    plt.axhline(y=F_2_haar, color='orange', linestyle='--',label = f'Ideal Haar distribution F = {F_2_haar:.3e}')
    plt.title(f'number of layers VS frame potenrial for n_qubits = {n_qubits}')
    plt.xlabel('number of layers')
    plt.ylabel('frame potential')
    plt.grid()
    plt.legend()
    plt.show()
    
    

    
# %%
