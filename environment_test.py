#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import traceback
import shutil

print("="*60)
print("🔍 PyTorch WSL2 快速诊断工具")
print("="*60)

# 1. 检查 Python 版本与路径
print("\n[1] Python 环境检查")
print("- Python 版本:", sys.version)
print("- Python 路径:", sys.executable)

# 2. 检查是否在 /mnt/c/ 目录下运行
cwd = os.getcwd()
if cwd.startswith("/mnt/"):
    print(f"- 当前路径在 Windows 挂载目录: {cwd}")
    print("  ⚠ 建议将代码移至 WSL 内部 ext4 文件系统 (如 ~/project) 提升稳定性")
else:
    print(f"- 当前路径在 WSL 内部目录: {cwd}")

# 3. 检查是否安装 PyTorch
print("\n[2] PyTorch 安装检查")
try:
    import torch
    print("- PyTorch 版本:", torch.__version__)
    print("- 安装路径:", torch.__file__)
except Exception as e:
    print("❌ 无法导入 torch:", e)
    sys.exit(1)

# 4. 检查 CUDA 是否可用
print("\n[3] CUDA 检查")
try:
    cuda_avail = torch.cuda.is_available()
    print("- CUDA 可用性:", cuda_avail)
    if cuda_avail:
        print("- GPU 数量:", torch.cuda.device_count())
        print("- 当前 GPU 名称:", torch.cuda.get_device_name(0))
    else:
        print("  ⚠ 未检测到 GPU，若需 GPU 需检查 WSL CUDA 驱动与 wsl --update")
except Exception:
    traceback.print_exc()

# 5. 检查线程库冲突
print("\n[4] 线程库检查 (MKL/OpenBLAS)")
env_mkl = os.environ.get("MKL_THREADING_LAYER")
print("- MKL_THREADING_LAYER =", env_mkl if env_mkl else "未设置")
try:
    test_code = """import torch, os; os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_THREADING_LAYER']='GNU'; import torch; print('OK')"""
    subprocess.run([sys.executable, "-c", test_code], timeout=10, check=True)
    print("  ✅ 设置 OMP_NUM_THREADS=1 + MKL_THREADING_LAYER=GNU 导入正常")
except subprocess.TimeoutExpired:
    print("  ❌ 导入 torch 超时，可能线程库冲突或文件系统卡死")

# 6. 检查是否多版本冲突
print("\n[5] 检查多版本 Torch 安装")
pip_show = shutil.which("pip3")
if pip_show:
    try:
        pip_list = subprocess.check_output([pip_show, "show", "torch"]).decode()
        print(pip_list)
    except subprocess.CalledProcessError:
        print("未通过 pip 安装 torch")
else:
    print("pip3 未找到")

# 7. 检查内核信息
print("\n[6] WSL 内核与系统信息")
print("- 平台:", platform.platform())
try:
    uname = subprocess.check_output(["uname", "-a"]).decode().strip()
    print("- uname:", uname)
except Exception:
    pass

print("\n诊断完成。")
print("="*60)
print("建议：")
print("1. 如果 CUDA 不可用且你安装了 GPU 版 PyTorch，请重装 CPU-only 版或配置 WSL CUDA")
print("2. 如果线程库冲突，尝试在代码中设置:")
print("   os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_THREADING_LAYER']='GNU'")
print("3. 如果在 /mnt/c/ 下运行，移动代码至 WSL 内部目录")
print("4. 确保 WSL2 与 Windows 驱动更新到最新 (wsl --update)")
print("="*60)
