import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm
        
class QFM():
    def __init__(self, input_samples, output_samples, n_ancilla=3, num_time_steps=3, depth_per_time_step=3):
        """Quantum Flow Model (QFM) for learning time evolution.

        Args:
            input_samples (array[float]): Array of shape (num_samples, dim_Hilbert_space) with initial states.
            output_samples (array[float]): Array of shape (num_samples, dim_Hilbert_space) with target states.
            num_time_steps (int): Number of time steps to model.
            depth_per_time_step (int): Depth of the PQC for each time step.
        """
        
        self.input_samples = input_samples
        self.output_samples = output_samples
        self.num_time_steps = num_time_steps
        self.depth_per_time_step = depth_per_time_step
        
        assert input_samples.shape[0] == output_samples.shape[0], "Input and output samples must have the same number of samples."
        assert input_samples.shape[1] == output_samples.shape[1], "Input and output samples must have the same number of features."

        self.n_input = int(np.log2(input_samples.shape[1]))
        self.n_output = int(np.log2(input_samples.shape[1]))

        self.n_ancilla = n_ancilla
        
        self.num_wires = 2 * self.n_input + self.n_ancilla + 1 # add ancilla qubits to approximate non-unitary evolution
        
        
        # Define wire indices
        self.wires_sys = range(self.n_input)
        self.wires_anc = range(self.n_input, self.n_input + self.n_ancilla)
        self.wires_all = range(self.n_input + self.n_ancilla)
        self.wires_target = range(self.n_input + self.n_ancilla, self.num_wires-1)
        self.swap_test_wires = [self.num_wires-1]  # Ancilla wire for swap test
        
        # Initialize parameters for the PQC
        self.parameters = np.random.normal(-0.1, 0.1, 
                                            (num_time_steps, depth_per_time_step, self.num_wires, 3), 
                                            requires_grad=True)
        
        # Define the quantum device
        self.dev = qml.device("lightning.qubit", wires=self.num_wires) #+n_input + 1  for swap-test
        
        # Define the quantum node of over all time steps
        @qml.qnode(self.dev, interface='autograd',  diff_method='adjoint')
        def circuit(params, x, y):
            '''
            Quantum circuit for a single time step.
            Args:
                params (array[float]): Parameters for the PQC.
                t (int): Current time step index.
                x (array[float]): Input state vector.
                y (array[float]): Target state vector.
            Returns:
                float: fidelity between the generated state and the target state.
            '''
            qml.templates.MottonenStatePreparation(x, wires=self.wires_sys)
            qml.templates.MottonenStatePreparation(y, wires=self.wires_target)
            for t in range(num_time_steps):
                self.PQC(params, time_step_index=t, depth=self.depth_per_time_step)
            
            # Swap test
            qml.Hadamard(wires=self.num_wires-1)
            for i in range(self.n_input):
                qml.CSWAP([self.num_wires-1, self.wires_sys[i], self.wires_target[i]])
            qml.Hadamard(wires=self.num_wires-1)
            
            result = qml.expval(qml.PauliZ(self.num_wires-1))
            return result
        self.circuit = circuit
        
        
    def PQC(self, parameters, time_step_index, depth=3):
        """Parameterized Quantum Circuit (PQC) for time evolution.

        Args:
            parameters (array[float]): Array of shape (depth, len(wires), 2) containing rotation angles.
            wires (list[int]): List of wire indices to apply the circuit on.
            time_step_index (int): Index of the current time step.
            depth (int): Number of layers in the PQC.
        """
        wires = range(self.n_input + self.n_ancilla)
        
        for depth_index in range(depth):
            for wire in wires:
                qml.RX(parameters[time_step_index, depth_index, wire, 0], wires=wire)
                qml.RY(parameters[time_step_index, depth_index, wire, 1], wires=wire)
                qml.RZ(parameters[time_step_index, depth_index, wire, 2], wires=wire)
            for i in range(1, len(wires)):
                qml.CNOT(wires=[wires[i], wires[i - 1]])
            qml.CNOT(wires=[wires[0], wires[-1]])
    
    def Interpolate(self, input, output, t_array):
        '''Limear interpolation between input and output states.
        Args:
            input (array[float]): Array of shape (num_samples, num_features) with initial states.
            output (array[float]): Array of shape (num_samples, num_features) with target states.
            t_array (array[float]): Array of time steps between 0 and 1.
        Returns:
            array[float]: Array of shape (len(t_array), num_samples, num_features) with interpolated states.
        '''
        
        if len(t_array) == 1:
            t_array = np.array([1.0])
        
        targets = np.zeros((len(t_array), input.shape[0], input.shape[1]))
        
        for i in range(len(t_array)):
            targets[i, :, :] = (1 - t_array[i]) * input + t_array[i] * output
            targets[i, :, :] = targets[i, :, :] / np.linalg.norm(targets[i, :, :], axis=1, keepdims=True)
        return targets
        
    
    def cost(self, params, t_index, x, y):
        
        
        @qml.qnode(self.dev, interface='autograd', diff_method='best')
        def ciruit(params):
            qml.templates.MottonenStatePreparation(x, wires=self.wires_sys)
                    
            qml.templates.MottonenStatePreparation(y, wires=self.wires_target)
            for _ in range(t_index + 1):
                self.PQC(params, time_step_index=_, depth=self.depth_per_time_step)
            
            # Swap test
            qml.Hadamard(wires=self.num_wires-1)
            for i in range(self.n_input):
                qml.CSWAP([self.num_wires-1, self.wires_sys[i], self.wires_target[i]])
            qml.Hadamard(wires=self.num_wires-1)
            
            result = qml.expval(qml.Identity(self.num_wires-1) - qml.PauliZ(self.num_wires-1))
            return result
        
        return ciruit(params)
    
    def fit(self, epochs=20, batch_size=5, learning_rate=0.01):
        """Train the QFM using the provided input and output samples.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            learning_rate (float): Learning rate for the optimizer.
        """
        opt = qml.QNGOptimizer() 
        #qml.AdamOptimizer(stepsize=learning_rate)
        
        # iterpolate targets for each time step
        num_samples = self.input_samples.shape[0]
        targets = self.Interpolate(self.input_samples, self.output_samples, np.linspace(0, 1, self.num_time_steps + 1, endpoint=True)[1::])
        
        # initialize input
        input = self.input_samples
        
        # Define cost function for optimization

        
        for t_index in range(self.num_time_steps):
            print(f"Training time step {t_index+1}/{self.num_time_steps}")
            input = self.input_samples
            target = targets[t_index]
            for epoch in range(epochs):
                # Shuffle the data
                indices = np.random.permutation(num_samples)
                input_shuffled = input[indices]
                target_shuffled = target[indices]
                
                for start in tqdm(range(0, num_samples, batch_size),
                                            desc=f"Epoch {epoch+1}/{epochs}",
                                            leave=False):
                    end = start + batch_size
                    x_batch = input_shuffled[start:end]
                    y_batch = target_shuffled[start:end]
                    
                    if len(x_batch) == 0:
                        continue  # skip empty batch
                    
                    x = np.mean(x_batch, axis=0) 
                    y = np.mean(y_batch, axis=0) 

                    # cost_fn = partial(self.circuit_per_time_step, t=t_index, x=x, y=y)
                    @qml.qnode(self.dev, interface='autograd', diff_method='best')
                    def cost(params):
                        qml.templates.MottonenStatePreparation(x, wires=self.wires_sys)
                        for _ in range(t_index + 1):
                            self.PQC(params, time_step_index=_, depth=self.depth_per_time_step)
                                                       
                        qml.templates.MottonenStatePreparation(y, wires=self.wires_target) 
                        
                        # Swap test
                        qml.Hadamard(wires=self.num_wires-1)
                        for i in range(self.n_input):
                            qml.CSWAP([self.num_wires-1, self.wires_sys[i], self.wires_target[i]])
                        qml.Hadamard(wires=self.num_wires-1)
                        
                        loss_operator = qml.Identity(self.num_wires-1) - qml.PauliZ(self.num_wires-1)
                        result = qml.expval(loss_operator)
                        return result

                    
 
                    self.parameters = opt.step(cost, self.parameters)
            
            print(f"Cost after time step {t_index+1}: {self.cost(self.parameters, t_index, np.mean(input,axis=0), np.mean(target,axis=0))}")
