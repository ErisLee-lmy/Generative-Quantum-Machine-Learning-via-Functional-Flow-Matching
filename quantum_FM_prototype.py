import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------
# 配置
# ---------------------
device = qml.device("default.mixed", wires=1)
epochs = 200
batch_size = 64
lr = 0.01
dtype = torch.float32

# ---------------------
# 辅助函数
# ---------------------
def bloch_to_angles(bloch):
    x, y, z = bloch
    theta = torch.acos(z.clamp(-1, 1))
    phi = torch.atan2(y, x)
    return theta, phi

def angles_to_bloch(theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def random_bloch_vectors(n):
    """均匀采样 Bloch 球表面"""
    u = torch.rand(n, dtype=dtype)
    v = torch.rand(n, dtype=dtype)
    theta = torch.acos(1 - 2 * u)
    phi = 2 * np.pi * v
    return angles_to_bloch(theta, phi).to(dtype)

def random_bloch_vectors_polar_cap(n, z_min=0.8):
    """采样北极帽区域 z >= z_min，保证返回 n 个点"""
    samples = []
    while len(samples) < n:
        vecs = random_bloch_vectors(n)
        mask = vecs[:, 2] >= z_min
        filtered = vecs[mask]
        if filtered.shape[0] > 0:
            if len(samples) == 0:
                samples = filtered
            else:
                samples = torch.cat([samples, filtered], dim=0)
    return samples[:n]

# ---------------------
# 速度场网络
# ---------------------
class QuantumVelocityField(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3)  # 输出速度
        )
    def forward(self, t, bloch):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([t.to(dtype), bloch.to(dtype)], dim=-1)
        return self.net(inp)

velocity_field = QuantumVelocityField()

# ---------------------
# 训练（速度场 MSE）
# ---------------------
optimizer = optim.Adam(velocity_field.parameters(), lr=lr)
loss_history = []

for epoch in range(epochs):
    # 初态 & 目标态
    x0 = random_bloch_vectors(batch_size)
    x1 = random_bloch_vectors_polar_cap(batch_size, z_min=0.8)

    # 时间
    t = torch.rand(batch_size, 1, dtype=dtype)

    # 插值位置
    xt = (1 - t) * x0 + t * x1

    # 目标速度
    v_star = x1 - x0  # 线性插值速度

    # 预测速度
    v_pred = velocity_field(t.squeeze(-1), xt)

    # MSE Loss
    loss = torch.mean((v_pred - v_star)**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

# ---------------------
# 保存Loss曲线
# ---------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
loss_path = os.path.join(script_dir, "Output", "quantum_flow_loss_linear_cap.png")
os.makedirs(os.path.dirname(loss_path), exist_ok=True)
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Velocity MSE Loss")
plt.title("Quantum Flow Matching (Linear Cap) Training Loss")
plt.savefig(loss_path)
print(f"Loss curve saved to {loss_path}")

# ---------------------
# 轨迹可视化（多起点）
# ---------------------
from mpl_toolkits.mplot3d import Axes3D

n_traj = 5  # 画5条轨迹
timesteps = np.linspace(0, 1, 20)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制 Bloch 球表面
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='lightblue', alpha=0.1)

# 绘制多条轨迹
with torch.no_grad():
    for _ in range(n_traj):
        x0_test = random_bloch_vectors(1)[0]
        xt = x0_test.clone()
        traj_points = []
        for ti in timesteps:
            v_pred = velocity_field(torch.tensor([ti], dtype=dtype), xt.unsqueeze(0)).squeeze(0)
            dt = 1.0 / len(timesteps)
            xt = xt + v_pred * dt
            xt = xt / torch.norm(xt)  # 归一化到Bloch球
            traj_points.append(xt.numpy())
        traj_points = np.array(traj_points)
        ax.plot(traj_points[:,0], traj_points[:,1], traj_points[:,2], 'r-', alpha=0.8)
        ax.scatter(traj_points[:,0], traj_points[:,1], traj_points[:,2], c='r', s=20)

# 目标区域标记（北极帽 z>0.8）
cap_points = random_bloch_vectors_polar_cap(100, z_min=0.8).numpy()
ax.scatter(cap_points[:,0], cap_points[:,1], cap_points[:,2], c='g', s=10, alpha=0.5, label="Target cap")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Quantum Flow Matching Trajectories (Polar Cap)')
ax.legend()

traj_path = os.path.join(script_dir, "Output", "bloch_trajectory_linear_cap.png")
plt.savefig(traj_path)
print(f"Bloch trajectory saved to {traj_path}")
