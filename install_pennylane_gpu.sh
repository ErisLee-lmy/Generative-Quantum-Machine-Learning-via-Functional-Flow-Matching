#!/bin/bash
set -e

echo "========= STEP 1: 卸载旧 CUDA ========="
sudo apt-get --purge remove "cuda*" -y || true
sudo apt-get autoremove -y
sudo apt-get update

echo "========= STEP 2: 添加 NVIDIA 官方 CUDA 12.1 仓库 ========="
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

echo "========= STEP 3: 安装 CUDA Toolkit 12.1 ========="
sudo apt-get install -y cuda-toolkit-12-1

echo "========= STEP 4: 配置环境变量 ========="
CUDA_PATH="/usr/local/cuda-12.1"
if ! grep -q "cuda-12.1" ~/.bashrc; then
    echo "export CUDA_HOME=$CUDA_PATH" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi
source ~/.bashrc

echo "========= STEP 5: 验证 CUDA 安装 ========="
nvcc --version || { echo "CUDA 安装失败"; exit 1; }

echo "========= STEP 6: 安装 Python 依赖 ========="
pip install --upgrade pip
pip install pennylane pennylane-lightning[gpu] cuquantum-python-cu12 custatevec-cu12

echo "========= 安装完成！请重启终端后运行检测脚本确认 ========="
