from math import pi
from qiskit import QuantumCircuit
import numpy as np
from QPE.HelpFunctions import bin_to_float, count_conversion, calculate_bitstring_distribution
from QPE.Unitary import Unitary
from QPE.PhaseEstimator import PhaseEstimator

def SuperIterativeCircuit(U, n_digits, states):
    dim = U.dim + 1
    N_states = len(states)
    n_measurents = (2**n_digits - 1) * N_states
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


class SuperIterative(PhaseEstimator):
    def __init__(self, U: Unitary, n_digits:int, N_states = 1, n_shots = 1024, backend_params: dict = {}, input_states = [[]], eigen_coefs = [[]]):
        super().__init__(U, n_digits, N_states, n_shots, backend_params, input_states, eigen_coefs)
        self.estimator_flag = "SuperIterative"
        
    def get_circuits(self):
        circ = [SuperIterativeCircuit(self.U, self.n_digits, self.states)]
        circ[0].draw("mpl").savefig("hallooo.jpg")
        return circ
    
    def post_process(self):
        counts = dict(self.job_results[0].get_counts())
        ones = count_conversion(counts)
        m_per_state = 2**self.n_digits - 1
        ones_per_state = [list(ones[m_per_state*i : m_per_state*(i+1)]) for i in range(self.N_states)]
        # new_order_ones_per_state = []
        for state_ones in ones_per_state:
            right_order_ones = state_ones[::-1]
            d = calculate_bitstring_distribution(right_order_ones, self.n_digits, self.n_shots)
            self.bitstring_distributions.append(d)
        estimated_bits = []
        for dist in self.bitstring_distributions:
            max_p = 0
            for key in dist:
                if dist[key] > max_p:
                    max_bitstring = key
                    max_p = dist[key]
            estimated_bits.append(max_bitstring)
        self.estimated_bits = estimated_bits
        self.estimated_phases = [bin_to_float(s) for s in self.estimated_bits]

if __name__ == "__main__":
    U = Unitary(random=True, random_state=123)
    eigen1 = [list(U.eigen[0].vector.copy())]
    print(eigen1)
    


    c = SuperIterativeCircuit(U, 4, eigen1)
    f = c.draw("mpl")
    f.savefig("SuperIterative.jpg")