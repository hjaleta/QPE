from re import U
from QPE.PhaseEstimator import PhaseEstimator
from QPE.Unitary import Unitary
from qiskit import QuantumCircuit
from math import pi
import numpy as np
from QPE.HelpFunctions import count_conversion, float_to_bin, bin_to_float
from typing import List

def KitaevCircuit(U, n_digits, states, re_initialize = True):
    dim = U.dim + 1
    N_states = len(states)
    N_class_regs = 2*N_states*(n_digits-2)
    circ = QuantumCircuit(dim, N_class_regs)
    circ.barrier()
    # circ.initialize([1,0], 0)
    target_qubits = list(range(1, dim))
    n_round = 0
    # class_reg = 0
    for state in states:
        for n in range(n_digits-2):
            for rot in range(2):
                circ.initialize([1,0], 0)
                if re_initialize or (n_round == 0):
                    circ.initialize(state, target_qubits)
                circ.barrier()
                circ.h(0)
                if rot == 1:
                    circ.rz(pi/2,0)
                power = 2**(n_digits - 3 - n)
                u = U.get_gate(power, 1)
                circ.append(u, list(range(dim)))
                circ.h(0)
                
                circ.measure(0,n_round)
                n_round += 1
                if n_round < N_class_regs: 
                    circ.barrier()

    return circ

class Kitaev(PhaseEstimator):
    def __init__(self, U:Unitary, n_digits:int, N_states:int = -1, input_states: List[list] = [[]], 
                eigen_coefs: List[list] = [[]], n_shots:int = 1024, backend_params: dict = {"service":"local"}, 
                method_specific_params ={}, re_initialize:bool = True,):

        if n_digits < 3:
            raise ValueError(f"Kitaevs algorithm estimates at least 3 digits, but n_digits={n_digits} was given")
        super().__init__(U=U, n_digits=n_digits, N_states=N_states, re_initialize=re_initialize,
                n_shots=n_shots, backend_params=backend_params, input_states=input_states,
                eigen_coefs=eigen_coefs, method_specific_params=method_specific_params)
        self.estimator_flag = "Kitaev"

    def get_circuits(self):
        circ = KitaevCircuit(self.U, self.n_digits, self.states)
        return [circ]

    def post_process(self):
        self.wait_for_job()
        result = self.jobs[-1].result()
        self.job_results.append(result)
        counts = np.flip(count_conversion(dict(result.get_counts())))

        ones = np.array(counts).reshape((self.N_states, self.n_digits-2, 2))

        rho = np.zeros((self.N_states, self.n_digits-2))
        for (N, n), _ in np.ndenumerate(rho):
            n1_cos = ones[N,n,0]
            n0_cos = self.n_shots - n1_cos
            cos = (n0_cos - n1_cos)/self.n_shots
            n1_sin = ones[N,n,1]
            n0_sin = self.n_shots - n1_sin
            sin = (n1_sin - n0_sin)/self.n_shots
            if cos == 0:
                if abs(sin-1) < 0.9:
                    angle = pi/2
                else:
                    angle = 3*pi/2
            else:
                angle = np.arctan(sin/cos)
                if cos < 0:
                    angle += pi
            if angle < 0:
                angle += 2*pi
            rho_Nn = angle/(2*pi)
            rho[N,n] = rho_Nn

        alpha = np.zeros((self.N_states, self.n_digits))

        for N in range(self.N_states):
            least_significant_rho = rho[N,0]
            #print(least_significant_rho)
            alpha_bits = float_to_bin(least_significant_rho, 3)
            # print(alpha_bits)
            alpha_bit_list = [int(b) for b in alpha_bits][::-1]
            # print(alpha_bit_list)
            alpha[N,0:3] = alpha_bit_list
            # print(N, alpha_bit_list)

        for N in range(self.N_states):
            for j in range(1, self.n_digits-2):
                rho_j = rho[N,j]
                # print(N,j,rho_j)
                alpha_eight = alpha[N,j]
                alpha_quarter = alpha[N,j+1]
                diff = rho_j - (alpha_quarter*1/4 + alpha_eight*1/8)
                if abs(diff) > 0.5:
                    diff = 1 - abs(diff)
                if diff <= 0.25:
                    alpha_half = 0
                else:
                    alpha_half = 1
                alpha[N, j+2] = alpha_half
        
        estimated_phases = []
        for N in range(self.N_states):
            bits = list(np.flip(alpha[N,:].flatten()))
            bitstring = "".join([str(int(bit)) for bit in bits])
            # print(bitstring)
            phase = bin_to_float(bitstring)
            estimated_phases.append(phase)
            
        self.estimated_phases = estimated_phases
        # estimated_bits = []
        bit_array = np.flip(alpha, axis = 1)
        for i in range(self.N_states):
            bit_list = [str(int(bit)) for bit in bit_array[i,:].copy()]
            bitstring = "".join(bit_list)
            self.estimated_bits.append(bitstring)

if __name__ == "__main__":
    from input_experiments import U2, U1
    from qiskit.visualization import plot_histogram
    u = Unitary(random= True, dim = 1)
    k = Kitaev(U2, 4, N_states=2, n_shots = 60)
    # u = Unitary(random= True, dim = 1)

    bit_dict0, bit_dict1 = {},{}

    # for i in range(60):
        
    k = Kitaev(U1, 4, N_states=2, n_shots = 60)
    k.run_circuit()
    k.post_process()
        # print(k.estimated_bits)
        # b0, b1 = k.estimated_bits
        # if b0 in bit_dict0:
        #     bit_dict0[b0] += 1
        # else:
        #     bit_dict0[b0] = 1
        # if b1 in bit_dict1:
        #     bit_dict1[b1] += 1
        # else:
        #     bit_dict1[b1] = 1
        
        # if i % 10 == 0:
        #     print(f"{i/10} percent done")

    # h0 = plot_histogram(bit_dict0, title="State 0")
    # h1 = plot_histogram(bit_dict1, title= "State 1")

    # h0.savefig("Kitaev_0_low_shots_60.png")
    # h1.savefig("Kitaev_1_low_shots_60.png")




    print(k.estimated_bits)
    print(k.U.get_phi_bitstrings(5))