import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn, optim
# ----------- 检查并选择设备 -----------
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- 量子设备（CPU）-----------
dev = qml.device("default.mixed", wires=1, shots=None)

# ----------- 单步演化电路 -----------
def step_circuit(theta, phi, gamma_step):
    """单个时间步：制备输入 -> 幅度阻尼"""
    qml.RY(theta, wires=0)
    qml.RZ(phi, wires=0)
    qml.AmplitudeDamping(gamma_step, wires=0)

# ----------- 多步演化 QNode -----------
@qml.qnode(dev, interface="torch")
def multi_step_qcircuit(thetas, phis, gammas):
    """
    thetas, phis: 初态制备角度
    gammas: 长度为N_steps的张量
    """
    # 初态制备
    qml.RY(thetas, wires=0)
    qml.RZ(phis, wires=0)

    # 多时间步演化
    for gamma_step in gammas:
        qml.AmplitudeDamping(gamma_step, wires=0)

    # 返回最终期望
    return qml.expval(qml.PauliZ(0))

# ----------- 量子模型类 -----------
class MultiStepQuantumModel(nn.Module):
    def __init__(self, n_steps=5):
        super().__init__()
        self.n_steps = n_steps
        # N个原始参数（无界），通过sigmoid限制到[0,1]
        self.raw_gammas = nn.Parameter(torch.zeros(n_steps))

    def forward(self, thetas, phis):
        gammas = torch.sigmoid(self.raw_gammas)  # 映射到[0,1]
        # 每个样本独立演化（注意转为CPU）
        expvals = [multi_step_qcircuit(t.cpu(), p.cpu(), gammas.cpu()) for t, p in zip(thetas, phis)]
        # 转回GPU
        return torch.stack(expvals).to(device), gammas

# ----------- 训练流程 -----------
def train_flow_matching(n_steps=5, epochs=200, batch_size=50, lr=0.05):
    model = MultiStepQuantumModel(n_steps=n_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # 均匀采样Bloch球（直接在GPU上生成）
        thetas = torch.acos(1 - 2 * torch.rand(batch_size, device=device))
        phis = 2 * np.pi * torch.rand(batch_size, device=device)

        # 前向计算
        expvals, gammas = model(thetas, phis)
        loss = torch.mean((1 - expvals) ** 2)

        # 反向优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Gammas={gammas.detach().cpu().numpy()}")

    # 测试
    test_thetas = torch.acos(1 - 2 * torch.rand(batch_size, device=device))
    test_phis = 2 * np.pi * torch.rand(batch_size, device=device)
    with torch.no_grad():
        final_expvals, final_gammas = model(test_thetas, test_phis)
    print("Final mean Z expectation:", torch.mean(final_expvals).item())
    print("Learned gammas per step:", final_gammas.detach().cpu().numpy())



# ----------- 运行 -----------
train_flow_matching(n_steps=5)
