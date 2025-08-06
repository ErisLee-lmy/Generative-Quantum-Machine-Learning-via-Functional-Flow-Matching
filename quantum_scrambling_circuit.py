import pennylane as qml
from pennylane import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as onp  # 标准 numpy 用于绘图
from mpl_toolkits.mplot3d import Axes3D
import os
# ==================================
# 1. 基本配置
# ==================================
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

steps = 5  # 扰乱步数
np.random.seed(42)  # 固定随机种子便于复现

# ==================================
# 2. 定义 QSC 层（随机旋转 + 纠缠）
# ==================================
def QSC_layer(params):
    """单层量子扰乱：随机旋转 + 纠缠"""
    # 单比特随机旋转
    for i in range(n_qubits):
        qml.RX(params[i, 0], wires=i)
        qml.RY(params[i, 1], wires=i)
        qml.RZ(params[i, 2], wires=i)
    # 纠缠门（相邻比特 CNOT）
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

# ==================================
# 3. 定义 QNode
# ==================================
@qml.qnode(dev)
def scrambled_state(initial_state, params_list):
    """从初始态开始，迭代应用多层 QSC"""
    qml.StatePrep(initial_state, wires=range(n_qubits))
    for params in params_list:
        QSC_layer(params)
    return qml.state()

# ==================================
# 4. Bloch 向量转换函数
# ==================================
def density_matrix_to_bloch(rho):
    """将单比特密度矩阵转为 Bloch 向量"""
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[1, 0])
    z = np.real(rho[0, 0] - rho[1, 1])
    return [x, y, z]

# ==================================
# 5. 前向扩散模拟
# ==================================
# 初始态 |000>
initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)

# 随机生成参数
param_steps = []
for _ in range(steps):
    params = 2 * np.pi * np.random.rand(n_qubits, 3)
    param_steps.append(params)

# 计算每一步的全系统量子态
states = []
for t in range(steps):
    state = scrambled_state(initial_state, param_steps[:t + 1])
    states.append(state)

# ==================================
# 6. 提取 Bloch 轨迹（qubit 0）
# ==================================
bloch_vectors = []
for state in states:
    # 计算全态密度矩阵
    rho_full = np.outer(state, np.conj(state))
    # 手动部分迹保留 qubit 0
    rho_qubit0 = np.zeros((2, 2), dtype=complex)
    for i in range(4):  # trace out qubit 1,2
        rho_qubit0[0, 0] += rho_full[i, i]       # |0xx>
        rho_qubit0[1, 1] += rho_full[i + 4, i + 4]  # |1xx>
        rho_qubit0[0, 1] += rho_full[i, i + 4]
        rho_qubit0[1, 0] += rho_full[i + 4, i]

    bloch_vec = density_matrix_to_bloch(rho_qubit0)
    bloch_vectors.append(bloch_vec)

# ==================================
# 7. 3D Bloch 球轨迹绘制
# ==================================
def plot_bloch_trajectory(vectors, title="Bloch sphere trajectory"):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 Bloch 球表面
    u, v = onp.mgrid[0:2*onp.pi:200j, 0:onp.pi:100j]
    x = onp.cos(u) * onp.sin(v)
    y = onp.sin(u) * onp.sin(v)
    z = onp.cos(v)
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.1, linewidth=0)

    # 坐标轴
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)

    # 绘制轨迹
    xs, ys, zs = zip(*vectors)
    ax.plot(xs, ys, zs, 'o-', color='black', markersize=5, label="Bloch vector path")

    # 设置范围和标签
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Output", "QSC_bloch_trajectory.png"))

# 绘制 Qubit 0 的轨迹
plot_bloch_trajectory(bloch_vectors, title="Qubit 0 Scrambling Trajectory")
