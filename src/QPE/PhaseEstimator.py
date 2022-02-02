import os
import sys
sys.path.insert(0,os.getcwd())
from qiskit import IBMQ, QuantumCircuit, QuantumRegister, transpile, execute
from qiskit.providers import backend
from qiskit.providers.jobstatus import JobStatus
from qiskit.visualization import plot_histogram
from Scripts.Backend import GetBackend, Login, filters
from Scripts.Circuits.Iterative import IterativeCircuit
from Scripts.Circuits.SuperIterative import SuperIterativeCircuit
from Scripts.Circuits.Kitaev import KitaevCircuit
from Scripts.Circuits.QFT import QFTCircuit
from Scripts.Circuits.AQFT import AQFTCircuit
from Scripts.Unitary import Unitary
from Scripts.HelpFunctions import bin_to_float, calculate_bitstring_distribution, float_to_bin, get_sub_bitstring_counter, phase_to_exp, count_conversion, mat_im_to_tuple
import numpy as np
import time
from math import pi, sqrt, sin, cos 
from typing import List
from datetime import datetime


class PhaseEstimator():

    """
    Class develloped in Qiskit with a goal to estimate Phases
    """

    def __init__(self, U:Unitary, n_digits:int, N_states:int = 1,
                 n_shots:int = 1024, backend_params: dict = {}, method_specific_params ={},
                 input_states: List[list] = [[]], eigen_coefs: List[list] = [[]]):

        self.U = U
        self.n_digits = n_digits
        self.method_specific_params = method_specific_params
        self.states, self.state_type, self.N_states = self.get_input_states(N_states, input_states, eigen_coefs)
        self.n_shots = n_shots
        self.backend_params = backend_params
        self.backend = self.get_backend()
        self.circuits = self.get_circuits()
        self.transpiled_circuits = self.get_transpiled_circuits()
        self.jobs = []
        self.job_results = []
        self.counts = []
        self.estimated_phases = []
        self.estimated_bits = np.zeros((self.N_states, self.n_digits))
        self.circuit_depths, self.transpiled_circuit_depths = [], []
        self.bitstring_distributions = []
        self.estimator_flag = ""


    def get_input_states(self, N_states, input_states, eigen_coefs):
        s = []
        if len(input_states[0]) == 0 and len(eigen_coefs[0]) == 0:
            if N_states > len(self.U.eigen):
                print(f"""Given value of 'N_states' is {N_states}, but Unitary has only {len(self.U.eigen)} eigenvectors.
                    Therefore 'N_states' is changed to {len(self.U.eigen)} """)
                N_states = len(self.U.eigen)
            elif N_states == -1:
                N_states = 2**self.U.dim
            for eig in self.U.eigen[:N_states]:
                s.append(eig.vector)
            state_type = "eigenvectors"
        elif len(input_states[0]) != 0:
            for state in input_states:
                if len(state) != 2**self.U.dim:
                    raise ValueError("Wrong size of input_state")
                
                s.append(state)
            state_type = "computational"
        elif len(eigen_coefs[0]) != 0:
            eigenvectors = self.U.eigen 
            for eigen_coef in eigen_coefs:
                if len(eigen_coef) != 2**self.U.dim:
                    raise ValueError("Wrong size of eigen_coefs")
                input_state = list(np.zeros(len(eigen_coef)))
                for i in range(len(input_state)):
                    new_eig = list(eigen_coef[i]*eigenvectors[i].vector)
                    input_state = [tot+new for tot,new in zip(input_state, new_eig)]
                norm = np.linalg.norm(input_state)
                input_state = input_state/norm
                s.append(input_state)
            state_type = "eigen combo"
        else:
            raise ValueError("No valid input to generate initial state")
        return s, state_type, len(s)

    def get_backend(self):
        service = self.backend_params["service"]
        if (service == "IBMQ" and IBMQ.active_account() == None) or service == "QI":
            print("YO")
            Login(service)
        backend = GetBackend(**self.backend_params)
        # print(backend.configuration().simulator)
        return backend

    def get_circuits(self):
        return [QuantumCircuit(1,1)]        

    def get_transpiled_circuits(self):
        transpiled_circuits = []
        for c in self.circuits:
            c_t = transpile(c, backend= self.backend, optimization_level=3)
            transpiled_circuits.append(c_t)
        return transpiled_circuits

    def run_circuit(self, index = -1):
        c_obj = self.transpiled_circuits[index]
        job = execute(c_obj, self.backend, shots = self.n_shots
        )
        self.jobs.append(job)
        self.job_results.append(job.result())

        self.wait_for_job()
        result = self.jobs[-1].result()
        self.job_results.append(result)

    def post_process(self):
        pass


    def get_depths(self):
        if len(self.circuits) == 0:
            raise ValueError("No circuits available")
        depths, transpiled_depths = [], []
        for circ, t_circ in zip(self.circuits, self.transpiled_circuits):
            d = circ.depth()
            t_d = t_circ.depth()
            depths.append(d)
            transpiled_depths.append(t_d)
        return depths, transpiled_depths        

    def get_status(self):
        if len(self.jobs) == 0:
            return "Job not submitted"
        else:
            job_status = self.jobs[-1].status()  # Query the backend server for job status.
            if job_status is JobStatus.QUEUED:
                pos = self.jobs[-1].queue_position()
                return f"Q {pos}"
            elif job_status == JobStatus.RUNNING:
                return "R"
            elif job_status == JobStatus.DONE:
                return "D"
            elif job_status == JobStatus.ERROR:
                return "E"
            elif job_status == JobStatus.VALIDATING:
                return "V"

    def wait_for_job(self, n_round = 0):
        start_time = time.time()
        max_minutes = 15
        new_status = self.get_status()
        if new_status == "D":
            waiting = False

        elif new_status == "Job not submitted":
            raise RuntimeError(f"Job not submitted")
        else:
            waiting = True
            while waiting:
                old_status = new_status
                time.sleep(2)
                new_status = self.get_status()
                if time.time() - start_time > 60*max_minutes:
                    raise RuntimeError(f"Script has been waiting for {max_minutes} minutes\nExecution terminated")
                # print(new_status)
                if new_status != old_status:
                    if new_status[0] == "Q":
                        print(f"In queue position {new_status[2]}")
                    elif new_status[0] == "R":
                        print("Job is running")
                    elif new_status[0] == "D":
                        print(f"Round {n_round} finished!")
                        waiting = False
                    elif new_status[0] == "E":
                        print("Error with job")
                        raise RuntimeError("Error with job")
                    elif new_status[0] == "V":
                        print("Validating job")

    def draw_circuits(self, path:str = "", N_non_transpiled:int = 1,
                     N_transpiled:int = 0, show = False):
        circs_to_draw = []
        if N_non_transpiled != 0:
            if (N_non_transpiled == -1) or (N_non_transpiled > len(self.circuits)):
                N_non_transpiled = len(self.circuits)
            circs_to_draw += self.circuits[:N_non_transpiled]
        if N_transpiled != 0:
            if (N_transpiled == -1) or (N_transpiled > len(self.transpiled_circuits)):
                N_transpiled = len(self.transpiled_circuits)
            circs_to_draw += self.transpiled_circuits[:N_transpiled]

        n_circuits = len(circs_to_draw)

        if path != "" and (len(path)>4) and (path[-4:] in [".jpg", ".png"]):
            path = path[:-4]
        for i in range(n_circuits):
            s = " - Non-transpiled" if i < N_non_transpiled else " - Transpiled"
            temp_path = path + s
            drawing = circs_to_draw[i].draw("mpl")
            if show:
                drawing.show()
            if n_circuits == 1:
                drawing.savefig(temp_path + ".jpg")
            else:
                drawing.savefig(temp_path + f" - Circuit {i}.jpg")
            

    def get_dicts(self):
        dicts = []
        u_dict = self.U.get_dict(self.n_digits)
        backend_dict = self.get_backend_dict()
        for i in range(self.N_states):
            input_state = mat_im_to_tuple(self.states[i])
            phase_dict = self.get_phase_dict(i)
            d = {
                "Estimator": self.estimator_flag,
                "Backend": backend_dict,
                "Unitary": u_dict,
                "Bits of precision": self.n_digits,
                "Input state": input_state,
                "estimated phase": phase_dict,
                "shots": self.n_shots,
                "timestamp": datetime.now().strftime("%m_%d_%Y_%H:%M")
            }
            dicts.append(d)
        return dicts

    def get_backend_dict(self):
        d = {}
        d["service"] = self.backend_params["service"]
        d["name"] = self.backend.name()
        d["n_qubits"] = self.backend.configuration().n_qubits
        d["simulator"] = self.backend.configuration().simulator
        return d

    def get_phase_dict(self, s_i):
        d = {}
        d["phi"] = self.estimated_phases[s_i]
        d["bitstring"] = self.bitstrings[s_i]
        if self.estimator_flag in ["Kitaev", "Iterative"]:
            d["distribution"] = {self.bitstrings[s_i]: 1}
        elif self.estimator_flag in ["QFT", "AQFT"]:
            d["distribution"] = {self.bitstrings[s_i]: 1}
        return d

    def get_bitstrings(self):
        # Return a list of the estimated bitstring orresponding to each state
        bitstrings = []
        if self.estimator_flag in ["Kitaev", "Iterative"]:
            for i in range(len(self.states)):
                b = "".join([str(int(bit)) for bit in self.estimated_bits[i,:].copy().flatten()])
                bitstrings.append(b)
        elif self.estimator_flag in ["QFT", "AQFT"]:
            pass
            #FIX THIS
        if len(bitstrings) != self.N_states:
            raise ValueError("Something is WRONG")
        return bitstrings
    
    def get_bitstring_distribution(self, s_i):
        if self.estimator_flag in ["Kitaev", "Iterative"]:
            return {
                self.bitstrings[s_i]: 1
            }
        elif self.estimator_flag in ["QFT", "AQFT"]:
            return self.counts[s_i]
            #FIX THIS

class Kitaev(PhaseEstimator):
    def __init__(self, U: Unitary, n_digits: int, N_states: int = 1, n_shots: int = 1024, backend_params: dict = {}, input_states =[[]], eigen_coefs = [[]] ):
        super().__init__(U, n_digits, N_states, n_shots, backend_params, input_states, eigen_coefs)
        self.estimator_flag = "Kitaev"

    def get_circuits(self):
        circ = KitaevCircuit(self.U, self.n_digits, self.states)
        return [circ]

    def post_process(self):
        self.wait_for_job()
        result = self.jobs[-1].result()
        self.job_results.append(result)
        counts = np.flip(count_conversion(dict(result.get_counts())))

        ones = np.array(counts).reshape((self.N_states, self.n_digits, 2))

        rho = np.zeros((self.N_states, self.n_digits))
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

        alpha = np.zeros((self.N_states, self.n_digits + 2))

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
            for j in range(1, self.n_digits):
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
        self.estimated_bits = np.flip(alpha, axis = 1)
        self.bitstrings = self.get_bitstrings()


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




class Iterative(PhaseEstimator):
    def __init__(self, U: Unitary, n_digits:int, N_states = 1, n_shots = 1024, backend_params: dict = {}, input_states = [[]], eigen_coefs = [[]]):
        super().__init__(U, n_digits, N_states, n_shots, backend_params, input_states, eigen_coefs)
        self.circuits = []
        self.estimator_flag = "Iterative"
        
    def get_one_circuit(self, n_round = 0, measurements = None):
        if not isinstance(measurements, type(np.zeros(0))):
            circ = QuantumCircuit(1,1)
        else:
            circ = IterativeCircuit(self.U, self.n_digits, 
                                n_round, measurements, self.states)
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
        for i in range(self.N_states):
            phase = 0
            bits = np.flip(self.measurements[i,:].copy())
            self.estimated_bits[i,:] = bits
            bit_list = list(bits.copy().flatten())
            bitstring = "".join([str(int(bit)) for bit in bit_list])
            phase = bin_to_float(bitstring)
            # for b_i, b in enumerate(bits):
            #     phase += 2**(- 1 - b_i)*b
            self.estimated_phases.append(phase)
        self.bitstrings = self.get_bitstrings()
  
class QFT(PhaseEstimator):
    def __init__(self, U: Unitary, n_digits: int, N_states: int = 1, n_shots: int = 1024, backend_params: dict = {}, input_states = [[]], eigen_coefs = [[]]):
        super().__init__(U, n_digits, N_states=N_states, n_shots=n_shots, backend_params=backend_params, input_states = input_states, eigen_coefs = eigen_coefs)
        self.estimator_flag = "QFT"

    def get_circuits(self):
        circ = QFTCircuit(self.U, self.n_digits, self.states)
        return [circ]
        # Return the required QFT Circuit

    def post_process(self):
        self.wait_for_job()
        # result = self.jobs[-1].result()
        # self.job_results.append(result)
        self.bitstring_distribution = []
        for r_i in range(len(self.job_results)):
            counts = dict(self.job_results[r_i].get_counts())
            b_dict = get_sub_bitstring_counter(counts, self.N_states)
            b_dict.reverse()
            self.bitstring_distribution.append(b_dict)
            estimated_bits = []
            for i in range(self.N_states):
                
                d = self.bitstring_distribution[r_i][i]
                maximum_counts = 0
                for key in d:
                    if d[key] > maximum_counts:
                        maximum_counts = d[key]
                        bitstring = key
                        # print("QFT")
                        # print(bitstring)
                estimated_bits.append(bitstring)
            
            estimated_bits.reverse()
            self.estimated_bits = estimated_bits
            self.estimated_phases = [bin_to_float(s) for s in self.estimated_bits]

class AQFT(PhaseEstimator):

    def __init__(self, U: Unitary, n_digits: int, N_states: int = 1, method_specific_params:dict = {"n_rotations":2},  
                    n_shots: int = 1024, backend_params: dict = {}, input_states = [[]], eigen_coefs = [[]]):

        super().__init__(U, n_digits, method_specific_params = method_specific_params,
                             N_states=N_states, n_shots=n_shots, backend_params=backend_params, 
                             input_states = input_states, eigen_coefs = eigen_coefs)
        self.estimator_flag = "AQFT"

    def get_circuits(self):
        n_controllers = self.method_specific_params["n_controllers"]
        circ = AQFTCircuit(self.U, self.n_digits, self.states, n_controllers)
        return [circ]
        # Return the required AQFT Circuit

def post_process(self):
        self.wait_for_job()
        # result = self.jobs[-1].result()
        # self.job_results.append(result)
        self.bitstring_distribution = []
        for r_i in range(len(self.job_results)):
            counts = dict(self.job_results[r_i].get_counts())
            b_dict = get_sub_bitstring_counter(counts, self.N_states)
            b_dict.reverse()
            self.bitstring_distribution.append(b_dict)
            estimated_bits = []
            for i in range(self.N_states):
                
                d = self.bitstring_distribution[r_i][i]
                maximum_counts = 0
                for key in d:
                    if d[key] > maximum_counts:
                        maximum_counts = d[key]
                        bitstring = key
                        # print("QFT")
                        # print(bitstring)
                estimated_bits.append(bitstring)
            
            estimated_bits.reverse()
            self.estimated_bits = estimated_bits
            self.estimated_phases = [bin_to_float(s) for s in self.estimated_bits]


if __name__ == "__main__":
    pass
    from Backend import Login, GetBackend, filters

    gods_phases1=np.array([9/16,9/16-1/50])
    diag_elements1 = phase_to_exp(gods_phases1)
    diag_matrix1 = np.diag(diag_elements1)
    U1=Unitary(diag_matrix1)

    gods_phases2=np.array([9/16-1/32,9/16-1/20])
    diag_elements2 = phase_to_exp(gods_phases2)
    diag_matrix2 = np.diag(diag_elements2)
    U2=Unitary(diag_matrix2)

    gods_phases3=np.array([0,1/3,1/69420,9/16])
    diag_elements3 = phase_to_exp(gods_phases3)
    diag_matrix3 = np.diag(diag_elements3)
    U3=Unitary(diag_matrix3)
    backend_params={ "service":"local" }#, "backend_name":"ibmq_qasm_simulator"}
    
    eigen_coefs = [
        [0,1]
    ]
    #QFT
    qft1=QFT(U1,6,2,backend_params=backend_params)
    # qft2=QFT(U2,4,2,backend_params= backend_params)
    # qft3_8=QFT(U1,4, eigen_coefs = eigen_coefs, backend_params= backend_params)
    # qft3_18=QFT(U3,18,4,backend_params= backend_params)

    # s = SuperIterative(U3,4, eigen_coefs=eigen_coefs, backend_params=backend_params)
    

    #AQFT
    # aqft1=AQFT(U1,6,2,backend_params=backend_params, n_controllers = 5)
    # aqft2=AQFT(U2,6,2,backend_params= backend_params, n_controllers = 3)
    # aqft3_8=AQFT(U3,4, eigen_coefs = eigen_coefs, backend_params= backend_params, n_controllers=2)
    # aqft3_18=AQFT(U3,18,4,backend_params= backend_params,n_controllers=17)

    # aqft1.draw_circuits(path="aqft.jpg")
    # qft1.draw_circuits(path="qft.jpg")

    # estimators = [qft3_8]#, aqft3_8, s]

    # for e in estimators:
    #     e.run_circuit()
    #     e.post_process()
    #     print(e.estimated_bits)
    #     print(e.estimated_phases)
    #     e.draw_circuits(path=f"{e.estimator_flag}.jpg")
    #     k = plot_histogram(e.job_results[0].get_counts())
    #     k.savefig(f"{e.estimator_flag} - histogram.jpg")
    #     # print(e.measurements)
        
    # phi3=U3.get_phis()
    # #phi3.reverse()
    # phi3_8_bin=[]
    # for j in range(0,len(phi3)):
    #     phi3_8_bin.append(float_to_bin(phi3[j],8))
    # print(phi3_8_bin)
    # print(phi3)
    
    # qft3_8.run_circuit()

    qft1.run_circuit()
    # aqft3_8.run_circuit()
    qft1.post_process()
    # aqft3_8.post_process()

    # print(qft3_8.estimated_bits)
    # print(qft3_8.estimated_phases)
    # print(aqft3_8.estimated_bits)
    # print(aqft3_8.estimated_phases)
