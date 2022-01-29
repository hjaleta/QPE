from math import pi
from qiskit import QuantumCircuit
import numpy as np
from Scripts.HelpFunctions import bin_to_float

def IterativeCircuit(U, n_digits, n_round, measurements, states):

    dim = U.dim + 1
    eigen = U.eigen
    N_states = len(states)
    circ = QuantumCircuit(dim,N_states)

    if measurements.shape[0] != N_states or measurements.shape[1] != n_digits:
        raise ValueError("Wrong dimensions of array 'measurements'")

    circ.initialize([1,0], 0)
    for s_i, state in enumerate(states):
        target_qubits = list(range(1, dim))
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
    
            

