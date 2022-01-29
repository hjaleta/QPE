from math import pi
from qiskit import QuantumCircuit
import numpy as np
from Scripts.HelpFunctions import bin_to_float
from Scripts.Unitary import Unitary

def SuperIterativeCircuit(U, n_digits, states):

    dim = U.dim + 1
    N_states = len(states)
    # print(dim)
    # print(n_digits)
    # print(N_states)
    n_measurents = (2**n_digits - 1) * N_states
    # print(n_measurents)
    circ = QuantumCircuit(dim, n_measurents)
    target_qubits = list(range(1, dim))
    
    classical_index = 0
    for s_i, state in enumerate(states):
        for k in range(n_digits):
            power = 2**(n_digits - 1 - k)
            n_shifts = 2**k
            u = U.get_gate(power, 1)
            for shift in range(n_shifts):
                circ.initialize(state, target_qubits)
                circ.initialize([1,0], 0)
                circ.barrier()
                circ.h(0)
                circ.append(u, list(range(dim)))
                angle_shift = shift*pi/n_shifts
                circ.rz(angle_shift,0)
                circ.h(0)
                circ.measure(0, classical_index)
                classical_index += 1
                if classical_index < n_measurents:
                    circ.barrier()
                    circ.reset(list(range(dim)))
    
    return circ

if __name__ == "__main__":
    U = Unitary(random=True, random_state=123)
    eigen1 = [list(U.eigen[0].vector.copy())]
    print(eigen1)
    


    c = SuperIterativeCircuit(U, 4, eigen1)
    f = c.draw("mpl")
    f.savefig("SuperIterative.jpg")