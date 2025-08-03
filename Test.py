import pennylane as qml
import torch
from torch import optim

# ----- 检查 lightning.gpu 是否可用 -----
try:
    dev = qml.device("lightning.gpu", wires=2)
    print("lightning.gpu backend is available.")
except Exception as e:
    raise RuntimeError(f"lightning.gpu backend not available: {e}")

# ----- 定义量子电路 -----
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(weights, x):
    # 编码输入数据
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    # 参数化旋转
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# ----- 简单数据（分类任务） -----
data = torch.tensor([[0.0, 0.0], [3.14, 0.0], [0.0, 3.14], [3.14, 3.14]], dtype=torch.float32)
labels = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32)  # 标签

# ----- 参数初始化 -----
weights = torch.nn.Parameter(torch.randn(2))

# ----- 优化器 -----
opt = optim.Adam([weights], lr=0.1)

# ----- 训练过程 -----
print("Training on GPU with lightning.gpu...")
for step in range(20):
    opt.zero_grad()
    preds = torch.stack([quantum_circuit(weights, x) for x in data])
    loss = torch.mean((preds - labels)**2)
    loss.backward()
    opt.step()
    if step % 5 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

print("Final weights:", weights)
