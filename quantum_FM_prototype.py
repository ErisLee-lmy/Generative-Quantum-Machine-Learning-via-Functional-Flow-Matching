# quantum_flow_matching.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt

# ---------------------
# 配置（自动检测 GPU & PennyLane 后端）
# ---------------------
torch_dtype = torch.float32
device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using torch device: {device_torch}")

# prefer lightning if available (may support GPU with proper install). fallback to default.qubit
qml_device_name = None
try:
    # try lightning.qubit first
    qml_device_name = "lightning.qubit"
    dev = qml.device(qml_device_name, wires=1)  # try create
    print(f"Using PennyLane device: {qml_device_name}")
except Exception:
    qml_device_name = "default.qubit"
    dev = qml.device(qml_device_name, wires=1)
    print(f"Falling back to PennyLane device: {qml_device_name}")

# hyperparams
epochs = 500
batch_size = 64
lr = 3e-3
pqc_layers = 3
latent_dim = 2 * pqc_layers  # per-layer paramization: RY + RZ per layer

# output folder
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
out_dir = os.path.join(script_dir, "Output")
os.makedirs(out_dir, exist_ok=True)

# ---------------------
# Bloch helpers (torch)
# ---------------------
def angles_to_bloch(theta: torch.Tensor, phi: torch.Tensor):
    # theta, phi shape (...,)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def random_bloch_vectors(n, dtype=torch_dtype, device=device_torch):
    u = torch.rand(n, dtype=dtype, device=device)
    v = torch.rand(n, dtype=dtype, device=device)
    theta = torch.acos(1 - 2 * u)
    phi = 2 * np.pi * v
    return angles_to_bloch(theta, phi).to(dtype)

def random_bloch_vectors_polar_cap(n, z_min=0.8, dtype=torch_dtype, device=device_torch):
    samples = []
    while len(samples) < n:
        need = n - len(samples)
        vecs = random_bloch_vectors(need*2, dtype=dtype, device=device)  # oversample
        mask = vecs[:, 2] >= z_min
        filtered = vecs[mask]
        if filtered.shape[0] > 0:
            samples.append(filtered)
    samples = torch.cat(samples, dim=0)
    return samples[:n]

# ---------------------
# Classical ParamNet (torch) — returns pqc params for each sample
# ParamNet input: [t, x,y,z] -> outputs (layers, 2) for RY and RZ angles
# ---------------------
class ParamNet(nn.Module):
    def __init__(self, hidden=128, layers=pqc_layers):
        super().__init__()
        self.layers = layers
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * layers)  # produce RY and RZ angles for each layer
        )
    def forward(self, t: torch.Tensor, bloch: torch.Tensor):
        # t: [batch,1]  bloch: [batch,3]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([t.to(torch_dtype).to(device_torch), bloch.to(torch_dtype).to(device_torch)], dim=-1)
        out = self.net(inp)  # [batch, 2*layers]
        return out.view(-1, self.layers, 2)  # [batch, layers, 2]

# ---------------------
# PQC (PennyLane QNode) maps params -> Bloch vector (expectation of Pauli's)
# We'll expose a helper that consumes a numpy or torch tensor of shape [layers,2] and returns [3]
# ---------------------
# build qnode
@qml.qnode(dev, interface="torch")
def quantum_circuit_torch(params):
    # params: shape [layers, 2] (each row: [ry, rz])
    # single qubit ansatz: chain of RY, RZ
    for i in range(params.shape[0]):
        qml.RY(params[i, 0], wires=0)
        qml.RZ(params[i, 1], wires=0)
    return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))

# ---------------------
# Hybrid module wrapping ParamNet + PQC
# ---------------------
class QuantumVelocityField(nn.Module):
    def __init__(self, layers=pqc_layers, hidden=128):
        super().__init__()
        self.param_net = ParamNet(hidden=hidden, layers=layers).to(device_torch)
        # PQC parameters are produced by ParamNet; QNode parameters are not trainable here.
    def forward(self, t: torch.Tensor, bloch: torch.Tensor):
        # t: [batch,1] ; bloch: [batch,3] -> output predicted velocity [batch,3]
        batch = t.shape[0]
        params_batch = self.param_net(t, bloch)  # [batch, layers, 2] on device_torch
        # We will collect PQC outputs per sample. If backend supports vectorized QNodes, replace loop.
        qouts = []
        for i in range(batch):
            # PennyLane QNode expects CPU tensors or compatible device depending on plugin.
            # We detach params to CPU if device not supported
            p = params_batch[i]
            try:
                bloch_pred = torch.stack(quantum_circuit_torch(p), dim=0)  # (3,)
            except Exception:
                # fallback: move to cpu numpy
                p_cpu = p.detach().cpu()
                px = quantum_circuit_torch(p_cpu)  # this returns torch tensor (on CPU)
                bloch_pred = torch.stack(px, dim=0)
            qouts.append(bloch_pred.unsqueeze(0))
        qouts = torch.cat(qouts, dim=0).to(device_torch)  # [batch,3]
        # velocity is a vector in Bloch coordinates; we can scale it (learnable scale) if needed
        return qouts

# ---------------------
# Instantiate model, optimizer
# ---------------------
model = QuantumVelocityField(layers=pqc_layers, hidden=128)
model.to(device_torch)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_hist = []

# ---------------------
# Training loop (Flow Matching style)
# - sample x0 (source) and x1 (target cap)
# - random t in (0,1), xt = (1-t)x0 + t x1
# - target velocity v* = x1 - x0  (flow matching uses this ground truth)
# - predict v_pred = v_theta(t, xt)
# - MSE loss between v_pred and v*
# ---------------------
print("Start training...")
for ep in range(epochs):
    # sample data on device
    x0 = random_bloch_vectors(batch_size, dtype=torch_dtype, device=device_torch)           # [batch,3]
    x1 = random_bloch_vectors_polar_cap(batch_size, z_min=0.8, dtype=torch_dtype, device=device_torch)  # [batch,3]
    t = torch.rand(batch_size, 1, dtype=torch_dtype, device=device_torch)  # [batch,1]
    xt = (1.0 - t) * x0 + t * x1
    v_star = (x1 - x0).to(device_torch)

    # forward
    model.train()
    v_pred = model(t, xt)  # [batch,3]

    loss = torch.mean((v_pred - v_star) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_hist.append(loss.item())
    if (ep % 50) == 0 or ep == epochs - 1:
        print(f"Epoch {ep:4d}/{epochs}: loss = {loss.item():.6f}")

# ---------------------
# 保存 Loss 曲线 & 可视化示例轨迹
# ---------------------
plt.figure()
plt.plot(loss_hist)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Quantum Flow Matching Loss')
loss_path = os.path.join(out_dir, "quantum_flow_loss.png")
plt.savefig(loss_path)
print(f"Saved loss curve to {loss_path}")

# 轨迹可视化：从噪声出发向 target cap 演化（用欧拉步长和模型预测速度）
from mpl_toolkits.mplot3d import Axes3D
n_traj = 5
timesteps = np.linspace(0, 1, 30)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
# Bloch sphere surface (light)
u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 40)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='lightblue', alpha=0.08)

with torch.no_grad():
    for _ in range(n_traj):
        x0_test = random_bloch_vectors(1, dtype=torch_dtype, device=device_torch)  # [1,3]
        x1_test = random_bloch_vectors_polar_cap(1, z_min=0.8, dtype=torch_dtype, device=device_torch)  # [1,3]
        xt = x0_test.clone()
        traj = []
        for ti in timesteps:
            t_tensor = torch.tensor([[ti]], dtype=torch_dtype, device=device_torch)
            v_pred = model(t_tensor, xt)  # [1,3]
            dt = 1.0 / len(timesteps)
            xt = xt + v_pred * dt
            # renormalize to Bloch sphere (keep pure-state length)
            xt = xt / torch.norm(xt, dim=-1, keepdim=True)
            traj.append(xt.squeeze(0).cpu().numpy())
        traj = np.array(traj)
        ax.plot(traj[:,0], traj[:,1], traj[:,2], '-', alpha=0.8)
        ax.scatter(traj[:,0], traj[:,1], traj[:,2], s=8)

cap_points = random_bloch_vectors_polar_cap(200, z_min=0.8, dtype=torch_dtype, device=device_torch).cpu().numpy()
ax.scatter(cap_points[:,0], cap_points[:,1], cap_points[:,2], c='g', s=6, alpha=0.5, label='Target cap')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.set_title('Quantum Flow Matching trajectories')
plt.legend()
traj_path = os.path.join(out_dir, "bloch_trajectories.png")
plt.savefig(traj_path)
print(f"Saved Bloch trajectories to {traj_path}")
