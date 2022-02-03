import os
import sys
sys.path.insert(0,os.getcwd())
from qiskit import IBMQ, QuantumCircuit, transpile, execute
from qiskit.providers.jobstatus import JobStatus
from QPE.Backend import GetBackend, Login
from QPE.Unitary import Unitary
from QPE.HelpFunctions import mat_im_to_tuple, mat_tuple_to_im
import numpy as np
import time
from typing import List, Union
from datetime import datetime
import json

class PhaseEstimator():

    """
    Superclass to algorithms for Quantum Phase Estimation

    Only one of the arguments 'N_states', 'input_states' or 'eigen_coefs' is needed

    Parameters
    ----------
    U: Unitary
        The unitary to perform phase estimation on
    
    n_digits: int
        The number of bits of precision the algorithm should use
    
    N_states: int
        The number of eigenstates to estimate the phase of. If parameter is set to -1, all states are estimated
    
    input_states: List[list]
        The input states that is initialized and estimated, in the computational basis. Each sublist should contain the complex coefficients that describe the desired statevector.
    
    eigen_coefs: List[list]
        The input states that is initialized and estimated, but in the eigenvector basis of the unitary. Each sublist should contain the complex coefficients that describe the desired statevector in the computational basis.

    n_shots: int
        Number of shots that we run the associated QuantumCircuit

    backend_params: dict
        Dictionary with kwargs to define the backend. See GetBackend in module QPE.Backend for description of these kwargs
    
    method_specific_params: dict
        Dictionary which hold parameters that are not common for all algorithms

    re_initialize: bool
        Parameter that decides whether the eigenstate should be re-inintialized between different trials or not

    Methods
    --------

    get_input_states:
        input: N_states, input_states, eigen_coefs
        output: A list of lists. Every sublist describes a state vector in the computational basis.
    
    get_backend:
        inp


    Attributes
    ----------
    U
    n_digits
    method_specific params    
    re_initialise

    states:
        A list of lists, where every sublist describes the input state in the computational basis.
        This 



    """


    def __init__(self, U:Unitary, n_digits:int, N_states:int = -1, input_states: List[list] = [[]], 
                eigen_coefs: List[list] = [[]], n_shots:int = 1024, backend_params: dict = {"service":"local"}, 
                method_specific_params ={}, re_initialize:bool = True,):
        self.U = U
        self.n_digits = n_digits
        self.method_specific_params = method_specific_params
        self.re_initialize = re_initialize
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
        self.estimated_bits = []
        self.circuit_depths, self.transpiled_circuit_depths = [], []
        self.bitstring_distributions = []
        self.estimator_flag = ""


    def get_input_states(self, N_states, input_states, eigen_coefs):
        s = []
        if ((len(input_states) == 1) and (len(input_states[0]) ==0)) and (len(eigen_coefs) == 1) and (len(eigen_coefs[0]) ==0):
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
        backend = GetBackend(**self.backend_params)
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
        # print("HALLAAA")
        start_time = time.time()
        max_minutes = 30
        status_print_period_minutes = 3
        print_time = time.time()
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
                current_time = time.time()
                if current_time - start_time > 60*max_minutes:
                    raise RuntimeError(f"Script has been waiting for {max_minutes} minutes\nExecution terminated")
                # print(new_status)
                if current_time - print_time > 60*status_print_period_minutes:
                    self.print_status(new_status)
                    print_time = current_time

                if new_status != old_status:
                    self.print_status(new_status)
                    
                    if new_status[0] == "D":
                        waiting = False

    def print_status(self, status):
        if status[0] == "Q":
            print(f"In queue position {status[2]}")
        elif status[0] == "R":
            print("Job is running")
        elif status[0] == "E":
            print("Error with job")
            raise RuntimeError("Error with job")
        elif status[0] == "D":
            print("Job done")
        elif status[0] == "V":
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
                "method_specific_params": self.method_specific_params,
                "re_initialize": self.re_initialize,
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
        # d["flags"] = self.backend_params
        return d

    def get_phase_dict(self, s_i):
        d = {}
        d["phi"] = self.estimated_phases[s_i]
        d["bitstring"] = self.estimated_bits[s_i]
        if self.estimator_flag in ["Kitaev", "Iterative"]:
            # print(self.e)
            d["distribution"] = {self.estimated_bits[s_i]: 1}
        elif self.estimator_flag == "QFT":
            d["distribution"] = self.bitstring_distributions[s_i]
        return d

    def get_bitstrings(self):
        # Return a list of the estimated bitstring orresponding to each state
        bitstrings = []
        if self.estimator_flag in ["Kitaev", "Iterative"]:
            for i in range(len(self.states)):
                b = "".join([str(int(bit)) for bit in self.estimated_bits[i,:].copy().flatten()])
                bitstrings.append(b)
        elif self.estimator_flag == "QFT":
            return self.estimated_bits
            #FIX THIS
        if len(bitstrings) != self.N_states:
            raise ValueError("Something is WRONG")
        return bitstrings
    
    def get_bitstring_distribution(self, s_i):
        if self.estimator_flag in ["Kitaev", "Iterative"]:
            return {
                self.bitstrings[s_i]: 1
            }
        elif self.estimator_flag == "QFT":
            return self.bitstring_distributions[s_i]
    
    def dump_to_json(self, path: Union[List[str], str]):
        dicts = self.get_dicts()
        n_dicts = len(dicts)
        if isinstance(path, str):
            if path[-5:] == ".json":
                path = path[:-5]
            paths = [f"{path}_state{s_i}.json" for s_i in range(n_dicts)]
        elif isinstance(path, list):
            paths = []
            for p in path:
                if (len(p) < 5) or (p[-5:] != ".json"):
                    p += ".json"
                paths.append(p)
        
        for d, p2 in zip(dicts, paths):
            with open(p2, "w") as write_file:
                json.dump(d, write_file, indent = True)
            print(f"Data saved to directory {p2}")

if __name__ == "__main__":
    pass