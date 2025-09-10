import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as qnp


class samples():
    def __init__(self, num_qubits_input, num_qubits_output, shots=100, backend='default.qubit'):
        self.n0 = num_qubits_input
        self.n1 = num_qubits_output
        self.d0 = 2 ** self.n0
        self.d1 = 2 ** self.n1
        
        self.shots = shots
        self.backend = backend

        self.dev = qml.device(self.backend, wires=self.num_qubits, shots=self.shots)
        

    def HaarStates(self, representation='vector'):
        if representation == 'vector':
            samples = []
            for _ in range(self.shots):
                state = qml.utils.random_state(self.d0)
                samples.append(state)
            return np.array(samples)

        elif representation == 'density_matrix':
            samples = []
            for _ in range(self.shots):
                state = qml.utils.random_state(self.d0)
                density_matrix = np.outer(state, np.conj(state))
                samples.append(density_matrix)
            return np.array(samples)

        else:
            raise ValueError("Representation must be 'vector' or 'density_matrix'")
        
        