from qiskit import QuantumCircuit
import numpy as np
from math import pi
from typing import List



def QFTCircuit(U, n_digits, input_states:List[list]):

    dim = U.dim + n_digits
    # eigen = U.get_eigen()
    N_states = len(input_states)
    circ = QuantumCircuit(dim,(dim-U.dim)*N_states)    

    for s_i, state in enumerate(input_states):
        circ.barrier()
        for i in range (n_digits):
            circ.initialize([1,0],i)

        # input state   
        target_qubits = list(range(n_digits, dim))
        circ.initialize(state, target_qubits)
        circ.barrier()
        circ.h(list(range(n_digits)))

        for i in range(n_digits):
            power = 2**i
            controller = n_digits -1 - i
            u = U.get_gate(power, ctrl = 1)
            circ.append(u, [controller]+list(range(n_digits, dim)))
        # circ.barrier()
        
        for i in range(0, n_digits):
            circ.h(i)
            # print(f"HADAMARD: {i}")
            for j in range(i+1,n_digits):
                angle = -pi*(2**(i-j))
                circ.crz(angle, i, j)
                # print(f"CTRL ROT: {i} -> {j}")

            # circ.h(i)
            #circ.barrier()

        for i in range (n_digits):
            circ.barrier()
            circ.measure(i,i+s_i*n_digits)
        
        #circ.measure_all()
        #circ.barrier()

    return circ













    # for i in range(N_states):
    #     eigenvector = eigen[i,1:]
    #     target_qubits = list(range(1, dim))
    #     circ.initialize(eigenvector, target_qubits)
    #     circ.h(0)
    #     previous_bits = np.flip(measurements[i,:n_round].copy())
    #     angle = 0
    #     for n, outcome in enumerate(previous_bits):
    #         angle += (-2*pi* 2**(- 2 - n))*outcome
    #     circ.rz(angle,0)
        
    #     power = 2**(n_digits-1-n_round)
    #     # print(power)
    #     u = U.get_gate(power, 1)
    #     circ.append(u, list(range(dim)))
    #     circ.h(0)
    #     circ.measure(0,i)
    #     if i < N_states - 1 :
    #         circ.barrier()
    #         circ.reset(list(range(dim)))
    #         # circ.initialize([1,0], 0)
    
    # return circ
    