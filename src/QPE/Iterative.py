from math import pi
from qiskit import QuantumCircuit, transpile, execute
import numpy as np
from QPE.HelpFunctions import bin_to_float, count_conversion
from QPE.PhaseEstimator import PhaseEstimator
from QPE.Unitary import Unitary
import time
from typing import List

def IterativeCircuit(U, n_digits, n_round, measurements, states, re_initialize):
    dim = U.dim + 1
    eigen = U.eigen
    N_states = len(states)
    circ = QuantumCircuit(dim,N_states)

    if measurements.shape[0] != N_states or measurements.shape[1] != n_digits:
        raise ValueError("Wrong dimensions of array 'measurements'")

    circ.initialize([1,0], 0)
    target_qubits = list(range(1, dim))
    for s_i, state in enumerate(states):

        circ.initialize(state, target_qubits)
        circ.h(0)
        previous_bits = list(np.flip(measurements[s_i,:n_round].copy().flatten()))
        previous_bit_string = [str(int(bit)) for bit in previous_bits]
        # print(previous_bit_string)
        phi_shift = - bin_to_float(previous_bit_string)/2
        angle_shift = 2*pi * phi_shift
        # for n, outcome in enumerate(previous_bits):
        #     angle += (-2*pi* 2**(- 2 - n))*outcome
        circ.rz(angle_shift,0)
        power = 2**(n_digits-1-n_round)
        # print(power)
        u = U.get_gate(power, 1)
        circ.append(u, list(range(dim)))
        circ.h(0)
        circ.measure(0, s_i)
        if s_i < N_states - 1 :
            circ.barrier()
            circ.reset(list(range(dim)))
    return circ
    
            
class Iterative(PhaseEstimator):
    def __init__(self, U:Unitary, n_digits:int, N_states:int = -1, input_states: List[list] = [[]], 
                eigen_coefs: List[list] = [[]], n_shots:int = 1024, backend_params: dict = {"service":"local"}, 
                method_specific_params ={}, re_initialize:bool = True,):

        super().__init__(U=U, n_digits=n_digits, N_states=N_states, re_initialize=re_initialize,
                n_shots=n_shots, backend_params=backend_params, input_states=input_states,
                eigen_coefs=eigen_coefs, method_specific_params=method_specific_params)

        self.circuits = []
        self.estimator_flag = "Iterative"
        
    def get_one_circuit(self, n_round = 0, measurements = None):
        if not isinstance(measurements, type(np.zeros(0))):
            circ = QuantumCircuit(1,1)
        else:
            circ = IterativeCircuit(self.U, self.n_digits, 
                                n_round, measurements, self.states, self.re_initialize)
        return circ

    def run_circuit(self):
        measurements = np.zeros((self.N_states, self.n_digits))
        for n_round in range(self.n_digits):
            circ = self.get_one_circuit(n_round, measurements)
            self.circuits.append(circ)
            transpiled_circuit = transpile(circ, backend=self.backend)
            self.transpiled_circuits.append(transpiled_circuit)
            self.jobs.append(execute(transpiled_circuit, self.backend, shots=self.n_shots))
            time.sleep(1) #We need to wait a bit, otherwise get_status=None
            self.wait_for_job(n_round=n_round)
            result = self.jobs[-1].result()
            self.job_results.append(result)
            counts = dict(result.get_counts())

            ones = count_conversion(counts)

            majority_bit_result = [1 if n_ones > self.n_shots/2 else 0 for n_ones in reversed(ones)]
            
            # print(majority_bit_result)
            measurements[:,n_round] = majority_bit_result
            # print(measurements)
        self.measurements = measurements
    
    def post_process(self):
        # estimated_bits = np.zeros((self.N_states, self.n_digits))
        for i in range(self.N_states):
            phase = 0
            bits = np.flip(self.measurements[i,:].copy())
            # estimated_bits[i,:] = bits
            bit_list = list(bits.copy().flatten())
            bitstring = "".join([str(int(bit)) for bit in bit_list])
            phase = bin_to_float(bitstring)
            # for b_i, b in enumerate(bits):
            #     phase += 2**(- 1 - b_i)*b
            self.estimated_phases.append(phase)
            self.estimated_bits.append(bitstring)

