# %%
"""
Quantum Flow Matching (QFM) — PennyLane + PyTorch (GPU-ready)
----------------------------------------------------------------
• Purpose: Give you a clean, fast scaffold to prototype a quantum version of flow matching.
• Highlights:
    - Auto GPU detection for both Torch and PennyLane (lightning.gpu if available; safe fallbacks).
    - Modular HWE ansatz (with optional ancilla) + time-conditioned gate controller (TimeNet).
    - QNode wired for torch autograd, adjoint/backprop selection based on device.
    - Mini training loop (teacher velocity = linear bridge baseline; swap-in your own).
    - Plenty of TODOs where you can drop in your quantum FM loss / Lindbladian, etc.

This file is meant to be edited as we iterate. Ping me with your constraints and I'll tailor it.
"""
from __future__ import annotations
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# ----------------------
# Global defaults
# ----------------------
DTYPE = torch.complex64  # complex32 compute; switch to complex128 for precision
REAL_DTYPE = torch.float32
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Utilities
# ----------------------

def try_make_pl_device(num_wires: int, shots: Optional[int] = None, prefer_gpu: bool = True):
    """Create the fastest available PennyLane device with graceful fallbacks.
    Order tried:
      1) lightning.gpu (CUDA/cuQuantum)
      2) lightning.qubit
      3) default.qubit.torch (keeps data on Torch device)
      4) default.qubit
    """
    # 1) lightning.gpu
    if prefer_gpu:
        try:
            dev = qml.device("lightning.gpu", wires=num_wires, shots=shots)
            return dev, "lightning.gpu"
        except Exception:
            pass
    # 2) lightning.qubit
    try:
        dev = qml.device("lightning.qubit", wires=num_wires, shots=shots)
        return dev, "lightning.qubit"
    except Exception:
        pass
    # 3) default.qubit.torch
    try:
        dev = qml.device("default.qubit.torch", wires=num_wires, shots=shots, dtype=REAL_DTYPE)
        return dev, "default.qubit.torch"
    except Exception:
        pass
    # 4) last resort: default.qubit
    dev = qml.device("default.qubit", wires=num_wires, shots=shots)
    return dev, "default.qubit"


def normalize_state(psi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(psi)
    return psi / (norm + eps)


def random_pure_states(batch: int, num_qubits: int, device: torch.device = TORCH_DEVICE) -> torch.Tensor:
    dim = 2 ** num_qubits
    real = torch.randn(batch, dim, dtype=torch.float32, device=device)
    imag = torch.randn(batch, dim, dtype=torch.float32, device=device)
    psi = torch.complex(real, imag)
    psi = torch.stack([normalize_state(v) for v in psi], dim=0).to(DTYPE)
    return psi

# ----------------------
# Ansatz: Hardware-Efficient with entangling layer
# ----------------------

def entangle_ring(wires):
    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def hwe_layer(theta: torch.Tensor, wires, use_rxryrz: bool = True):
    """One HWE layer.
    theta shape: (..., len(wires), 3) if RX/RY/RZ; else (..., len(wires), 1)
    """
    if use_rxryrz:
        for i, w in enumerate(wires):
            a, b, c = theta[..., i, 0], theta[..., i, 1], theta[..., i, 2]
            qml.RX(a, wires=w)
            qml.RY(b, wires=w)
            qml.RZ(c, wires=w)
    else:
        for i, w in enumerate(wires):
            qml.RY(theta[..., i, 0], wires=w)
    entangle_ring(wires)

# ----------------------
# Time-conditioned controller for gate angles
# ----------------------
class TimeNet(nn.Module):
    def __init__(self, num_qubits: int, hidden: int = 128, use_rxryrz: bool = True):
        super().__init__()
        self.num_qubits = num_qubits
        self.use_rxryrz = use_rxryrz
        out_dims = num_qubits * (3 if use_rxryrz else 1)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dims),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: shape (B, 1), values in [0,1]. Returns angles shaped (B, num_qubits, K)."""
        angles = self.net(t)
        k = 3 if self.use_rxryrz else 1
        angles = angles.view(-1, self.num_qubits, k)
        return angles

# ----------------------
# Quantum Flow Module
# ----------------------
class QuantumFlowMatching(nn.Module):
    def __init__(
        self,
        num_qubits: int,
        num_ancilla: int = 0,
        layers_per_step: int = 2,
        time_steps: int = 8,
        prefer_gpu_device: bool = True,
        shots: Optional[int] = None,
        use_rxryrz: bool = True,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_ancilla = num_ancilla
        self.total_wires = num_qubits + num_ancilla
        self.layers_per_step = layers_per_step
        self.time_steps = time_steps
        self.use_rxryrz = use_rxryrz

        self.dev, self.backend_name = try_make_pl_device(self.total_wires, shots=shots, prefer_gpu=prefer_gpu_device)

        # Time-conditioned controller
        self.controller = TimeNet(self.total_wires, hidden=192, use_rxryrz=use_rxryrz)

        # Choose diff method: adjoint for statevector sims; backprop for torch device
        if self.backend_name in ("lightning.qubit", "lightning.gpu", "default.qubit"):
            diff_method = "adjoint"
        else:
            diff_method = "backprop"

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def evolve_qnode(state_in: torch.Tensor, t_scalar: torch.Tensor):
            """Apply L discretized layers at time t.
            state_in: complex vector (2^(total_wires),)
            t_scalar: shape () in [0,1]
            """
            qml.QubitStateVector(state_in, wires=range(self.total_wires))
            # Repeat a few layers at this time slice
            for _ in range(self.layers_per_step):
                # Broadcast controller angles for a batch of size 1 at time t
                angles = self.controller(t_scalar.view(1, 1))  # (1, total_wires, K)
                hwe_layer(angles[0], wires=range(self.total_wires), use_rxryrz=self.use_rxryrz)
            return qml.state()

        self.evolve_qnode = evolve_qnode

    def forward(self, psi0: torch.Tensor) -> torch.Tensor:
        """Discretized time evolution from t=0 to t=1.
        psi0: (B, 2^n) complex state vectors on the active device.
        Returns psi1_pred: (B, 2^n)
        """
        B, dim = psi0.shape
        psi = psi0
        # Uniform grid in [0,1)
        ts = torch.linspace(0.0, 1.0 - 1.0 / self.time_steps, self.time_steps, dtype=torch.float32, device=psi0.device)
        for t in ts:
            # Evolve each batch element independently (torch vmap-like loop)
            next_psi = []
            for b in range(B):
                out_state = self.evolve_qnode(psi[b], t)
                next_psi.append(out_state)
            psi = torch.stack(next_psi, dim=0)
            # Optional: renorm to fight drift
            psi = torch.stack([normalize_state(v) for v in psi], dim=0)
        return psi

# ----------------------
# Simple FM-style training objective (baseline)
# ----------------------
class FMObjective(nn.Module):
    """A placeholder Flow Matching objective on amplitudes.

    We use a linear bridge between amplitudes: x_t = (1 - t) x0 + t x1.
    Teacher velocity: v*(x_t, t) = x1 - x0 (constant). We supervise the model's instantaneous
    velocity via finite difference of the model flow. This is a crude baseline; replace with
    your preferred quantum FM target (e.g., score-based, Lindbladian velocity, etc.).
    """
    def __init__(self, time_steps: int, fd_epsilon: float = 1e-2):
        super().__init__()
        self.time_steps = time_steps
        self.fd_eps = fd_epsilon

    def forward(self, model: QuantumFlowMatching, psi0: torch.Tensor, psi1: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B = psi0.shape[0]
        device = psi0.device
        # Random time t ~ U[0,1)
        t = torch.rand(B, 1, device=device)
        # Linear bridge on amplitudes (demo only)
        x_t = (1.0 - t) * psi0 + t * psi1
        x_t = torch.stack([normalize_state(v) for v in x_t], dim=0)
        # One model step from x_t  (reuse the model but with 1 time-slice by setting time_steps=1 ad-hoc)
        # We simulate a tiny step by calling the qnode once at time t with a single layer set.
        # Create a shallow copy with 1 step without breaking autograd
        old = model.time_steps
        model.time_steps = 1
        with torch.enable_grad():
            pred_xt_next = []
            for b in range(B):
                pred = model.evolve_qnode(x_t[b], t[b, 0])
                pred_xt_next.append(pred)
            pred_xt_next = torch.stack(pred_xt_next, dim=0)
        model.time_steps = old

        # Finite-difference velocity of the model around x_t
        v_model = (pred_xt_next - x_t)  # Δt ≈ 1 step; treat as proportional to velocity
        v_teacher = (psi1 - psi0)

        loss = F.mse_loss(v_model.real, v_teacher.real) + F.mse_loss(v_model.imag, v_teacher.imag)
        metrics = {"mse_real": F.mse_loss(v_model.real, v_teacher.real).item(),
                   "mse_imag": F.mse_loss(v_model.imag, v_teacher.imag).item()}
        return loss, metrics

# ----------------------
# Training harness
# ----------------------
class Trainer:
    def __init__(self, model: QuantumFlowMatching, lr: float = 3e-4):
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
        try:
            self.model = torch.compile(self.model, fullgraph=False, dynamic=True)  # PyTorch 2.x
        except Exception:
            pass

    def fit(self, data_iter, steps: int = 200):
        obj = FMObjective(time_steps=self.model.time_steps)
        for step in range(1, steps + 1):
            psi0, psi1 = next(data_iter)
            psi0 = psi0.to(TORCH_DEVICE).to(DTYPE)
            psi1 = psi1.to(TORCH_DEVICE).to(DTYPE)

            self.opt.zero_grad(set_to_none=True)
            loss, metrics = obj(self.model, psi0, psi1)
            loss.backward()
            self.opt.step()

            if step % 10 == 0:
                print(f"step {step:04d} | loss={loss.item():.6f} | backend={self.model.backend_name} | metrics={metrics}")

# ----------------------
# Example synthetic dataloader
# ----------------------
class ToyQuantumPairs:
    """Yield batches of (psi0, psi1) where psi1 = random unitary * psi0.
    This is just to wire up the system; replace with your real dataset / target sampler.
    """
    def __init__(self, batch: int, num_qubits: int, steps_per_epoch: int = 100):
        self.batch = batch
        self.num_qubits = num_qubits
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.steps_per_epoch:
            self.i = 0
        self.i += 1
        B = self.batch
        psi0 = random_pure_states(B, self.num_qubits, TORCH_DEVICE)
        # Create psi1 via a shallow HWE layer (pseudo random unitary)
        dim = 2 ** self.num_qubits
        # For now, just draw another random state (hard target)
        psi1 = random_pure_states(B, self.num_qubits, TORCH_DEVICE)
        return psi0, psi1

# ----------------------
# Entry point (manual test)
# ----------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    NUM_QUBITS = 4
    ANCILLA = 0  # set >0 to append ancillas (they're just carried along in the flow for now)
    BATCH = 16

    model = QuantumFlowMatching(
        num_qubits=NUM_QUBITS,
        num_ancilla=ANCILLA,
        layers_per_step=2,
        time_steps=6,
        prefer_gpu_device=True,
        shots=None,
        use_rxryrz=True,
    ).to(TORCH_DEVICE)

    data = iter(ToyQuantumPairs(batch=BATCH, num_qubits=NUM_QUBITS, steps_per_epoch=10))
    trainer = Trainer(model, lr=2e-3)
    trainer.fit(data, steps=50)

    # TODOs for you to customize next:
    # 1) Replace FMObjective with your quantum FM teacher velocity (e.g., Schrödinger/Lindblad).
    # 2) Add ancilla + partial trace to implement noisy channels; measure-out ancilla if desired.
    # 3) Swap ToyQuantumPairs with (psi0, psi1) sampled from your target model (e.g., TFIM ground states).
    # 4) Add checkpointing, AMP (autocast) for large models, mixed precision if/when stable.
# %%