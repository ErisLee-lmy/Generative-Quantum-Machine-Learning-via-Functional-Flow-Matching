#!/bin/bash
echo "==================== GPU & CUDA CHECK ===================="

# 检查 CUDA Toolkit
if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA Toolkit detected:"
    nvcc --version | grep "release"
else
    echo "ERROR: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

# 检查 CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
    echo "WARNING: CUDA_HOME not set, attempting default path (/usr/local/cuda)"
    export CUDA_HOME=/usr/local/cuda
else
    echo "CUDA_HOME=$CUDA_HOME"
fi

# 检查 NVIDIA 驱动和 GPU
echo "-------------------- nvidia-smi --------------------"
nvidia-smi || { echo "ERROR: No NVIDIA GPU detected!"; exit 1; }

echo "==================== PYTHON ENV CHECK ===================="
python3 - <<'EOF'
import sys
try:
    import pennylane as qml
except ImportError:
    sys.exit("ERROR: PennyLane not installed.")
try:
    from cuquantum import custatevec
except ImportError:
    print("WARNING: cuQuantum not found or not imported, check installation.")

# 列出可用设备
print("Available PennyLane devices:", qml.devices.available())

# 测试 lightning.gpu
if "lightning.gpu" not in qml.devices.available():
    sys.exit("ERROR: lightning.gpu backend not available. Check installation.")

# 测试运行简单电路
dev = qml.device("lightning.gpu", wires=2)
@qml.qnode(dev)
def circuit(x):
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

result = circuit([0.5, 0.1])
print("Test circuit result:", result)
print("Device capabilities:", dev.capabilities())
EOF

echo "==================== GPU USAGE SNAPSHOT ===================="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader

echo "Check complete."
