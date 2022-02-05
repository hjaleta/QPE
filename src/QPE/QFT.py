from QPE.PhaseEstimator import PhaseEstimator
from QPE.Unitary import Unitary
from qiskit import QuantumCircuit
from math import pi
import numpy as np
from QPE.HelpFunctions import count_conversion, float_to_bin, get_sub_bitstring_counter, bin_to_float
from typing import List
from qiskit.visualization import plot_histogram

def QFTCircuit(U, m_digits, input_states:List[list], n_rotations = 2, re_initialize = True):

    dim = U.dim + m_digits
    N_states = len(input_states)
    circ = QuantumCircuit(dim,(dim-U.dim)*N_states)

    if (n_rotations > m_digits - 1) or n_rotations == -1:
        n_rotations = m_digits -1

    for s_i, state in enumerate(input_states):
        circ.barrier()
        for i in range (m_digits):
            circ.initialize([1,0],i)
      
        target_qubits = list(range(m_digits, dim))
        circ.initialize(state, target_qubits)
        circ.barrier()
        circ.h(list(range(m_digits)))

        for i in range(m_digits):
            power = 2**i
            controller = m_digits -1 - i
            u = U.get_gate(power, ctrl = 1)
            circ.append(u, [controller]+list(range(m_digits, dim)))

        for i in range(0, m_digits):
            circ.h(i)
            for j in range(i+1, i + n_rotations + 1):
                if j<(m_digits):
                    angle = -pi*(2**(i-j))
                    circ.crz(angle, i, j)

        for i in range (m_digits):
            circ.barrier()
            circ.measure(i,i+s_i*m_digits)
    return circ

class QFT(PhaseEstimator):
    """
    SUbclass of th PhaseEstimator class

    Args:
        PhaseEstimator ([type]): [description]
    """
    def __init__(self, U:Unitary, m_digits:int, N_states:int = -1, input_states: List[list] = [[]], 
                eigen_coefs: List[list] = [[]], n_shots:int = 1024, backend_params: dict = {"service":"local"}, 
                method_specific_params ={"n_rotations":-1}, re_initialize:bool = True,):
        
        super().__init__(U=U, m_digits=m_digits, N_states=N_states, re_initialize=re_initialize,
                n_shots=n_shots, backend_params=backend_params, input_states=input_states,
                eigen_coefs=eigen_coefs, method_specific_params=method_specific_params)
        
        self.estimator_flag = "QFT"

    def get_circuits(self):
        circ = QFTCircuit(self.U, self.m_digits, self.states, self.method_specific_params["n_rotations"])
        return [circ]
        # Return the required QFT Circuit

    def post_process(self):
        self.wait_for_job()
        self.bitstring_distributions = []
        #for r_i in range(len(self.job_results)):
        counts = dict(self.job_results[-1].get_counts())
        b_dict = get_sub_bitstring_counter(counts, self.N_states)
        b_dict.reverse()
        #print(b_dict)
        self.bitstring_distributions = b_dict
        estimated_bits = []
        for i in range(self.N_states):
            d = self.bitstring_distributions[i]
            maximum_counts = 0
            for key in d:
                if d[key] > maximum_counts:
                    maximum_counts = d[key]
                    bitstring = key
            estimated_bits.append(bitstring)
        
        estimated_bits.reverse()
        self.estimated_bits = estimated_bits
        self.estimated_phases = [bin_to_float(s) for s in self.estimated_bits]
    
    def plot_histograms(self, s_i):
        d = self.bitstring_distributions[s_i]
        # title = fr"Input state $\varphi_{s_i}$"
        x = [key for key in d]
        n = [d[key] for key in d]
        hist = plot_histogram(d)
        return hist
        # print(n)
        # plt.hist(d)
        # plt.show()
        # fig.show()
        # return fig
