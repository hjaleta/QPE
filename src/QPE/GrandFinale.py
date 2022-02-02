from QPE.PhaseEstimator import PhaseEstimator, Iterative, Kitaev, QFT, AQFT
from QPE.HelpFunctions import phase_to_exp
from Unitary import Unitary, Eigenvector
from typing import List
import json
import numpy as np
from QPE.Backend import GetBackend, filters
from datetime import datetime
import os
import re

class ExperimentSet():

    def __init__(self, instructions, path, draw_params:dict = {"NT":0, "T":0}):
        self.instructions = instructions
        self.path = path
        self.draw_params = draw_params
        self.name, self.filename_flags = self.get_name_and_filename_flags()
        self.PE_dicts = self.get_PE_dicts()
        self.generate_filenames()
        self.result_dicts = []
        
    def get_name_and_filename_flags(self):
        name = self.instructions.pop("name")
        filename_flags = self.instructions.pop("filename_flags")
        return name, filename_flags

    def get_PE_dicts(self):
        PE_dicts = [
            {
                # "experiment name": self.name
            }
        ]
        non_defaults = list(self.instructions.keys())

        def generate_combinations(PE_dicts, key, param_list):
            new_PE_dicts = []
            for old_dict in PE_dicts:
                for param in param_list:
                    new_d = old_dict.copy()
                    new_d[key] = param
                    new_PE_dicts.append(new_d)
            return new_PE_dicts

        while len(non_defaults) > 0:
            key = non_defaults.pop(-1)
            param_list = self.instructions[key]
            PE_dicts = generate_combinations(PE_dicts, key, param_list)

        return PE_dicts

    def generate_filenames(self):

        keys = list(self.PE_dicts[0].keys())
        estimator_pattern = r"[A-Za-z]+"
        estimator_pattern = r"<class 'PhaseEstimator\.([A-Za-z]+)'>"

        print(keys)
        print(self.filename_flags)
        for d in self.PE_dicts:
            filename = ""
            for flag in self.filename_flags:
                if flag == "estimator":
                    full_e_string = str(d["estimator"])
                    # print(type(full_e_string))
                    # print(full_e_string)
                    match = re.search(estimator_pattern, full_e_string)
                    s = match.groups()[0]
                    filename += s
                elif flag == "U":
                    unitary_name = d["U"].name
                    if unitary_name != "":
                        filename += f"_{unitary_name}"
                elif flag == "backend_name":
                    filename += f"_{d['backend_params']['backend_name']}"
                elif flag == "backend_service":
                        filename += f"_{d['backend_params']['service']}"
                        # print("WOW")
                elif flag == "method_specific_params":
                    if "n_rotations" in d["method_specific_params"]:
                        filename += f"_n_rotations{d['method_specific_params']['n_rotations']}" 
                elif flag in ["n_digits","n_shots"]:
                    filename += f"_{flag}{d[flag]}"                   

            d["filename"] = filename



    def one_run(self, index:int, draw_params = {"T":-1, "NT":-1}):
        PE_dict = self.PE_dicts[index]
        filename = PE_dict.pop("filename")
        try:
            estimator = PE_dict.pop("estimator")
            e = estimator(**PE_dict)
            e.run_circuit()
            e.post_process()
        except:
            raise RuntimeError("Something went wrong with the PhaseEstimator")
        
        if (draw_params["T"] != 0) or (draw_params["NT"] != 0):
            plot_path = self.path + f"/plots/{filename}"
            if not os.path.isdir(plot_path):
                os.makedirs(plot_path, exist_ok=True)
                print(f"Created directory {plot_path}")
            
            drawing_path = plot_path + f"{filename}_{index}"
            # print(drawing_path)
            e.draw_circuits(drawing_path, self.draw_params["NT"], self.draw_params["T"])

        dicts = e.get_dicts()
        json_path = self.path + "/data"

        if not os.path.isdir(json_path):
            os.mkdir(json_path)
            os.makedirs(json_path, exist_ok=True)
            print(f"Created directory {json_path}")

        for d_i, d in enumerate(dicts):
            d["Experiment"] = self.name
            d["name"] = filename
            json_filepath = json_path + f"/{filename} - {index}_{d_i+1}.json"
            with open(json_filepath, "w") as write_file:
                json.dump(d, write_file, indent = True)
        
    def run(self):
        for i in range(len(self.PE_dicts)):
            # try:
            self.one_run(i)
            # except:
            #     print(f"Experiment {self.name} failed for PhaseEstimator described by paramaters {self.PE_dicts[i]}")



if __name__ == "__main__":

    # estimators = [Kitaev, Iterative]
    diag_matrix = np.diag(phase_to_exp([0.3, 0.85]))
    unitaries = [
        Unitary(random=True, random_state=123, dim = 1),
        # Unitary(diag_matrix)
        ]
    n_digits_list = list(range(2,4))
    backend_param_list = [
        {
            "service": "local"
        }
        # ,
        # {
        #     "service": "IBMQ", 
        #     "filters": filters["5qubit"]
        # }
    ]
    state_dicts = [
        {"N_states": 2},
        {"eigen_coefs": [
            [1,1],
            [3,1]
        ]}
    ]

    instructions = {
    "name": "superexperiment",
    "estimator": [Kitaev, Iterative],

    "U": [Unitary(random=True, random_state=123, dim = 1)],
    "n_digits": list(range(2,4)),
    "backend_params": [{"service": "local"}],
    "n_shots": [500],
    "N_states": [2],
    "filename_flags": ["estimator", "backend_service", "n_digits"]
    }

    path = "Results/Experiments/superexperiment"

    
    superexperiment = ExperimentSet(instructions, path)
    superexperiment.run()

# j = {
    
#     "Estimator": "Kitaev",
#     "Backend":{
#         "host": "local",
#         "simulator": True,
#         "nqubits": 5,
#         "name": "lima",
#     },

#     "Unitary": {
#         "dimension": 1,
#         "data": [
#             [(0.3, 0.6)],
#             []
#         ],
#         "eigenvectors":[
#             [(1,0),(0,0)],
#             [(1,0.5),(0.1,0.2)]
#         ],
#         "phis":[],
#         "phi_bitstrings": []
#     },

#     "Bits of precision": 3,
#     "Input state": [0,0.5,0.5,0],
    
#     "estimated phase": {
#         "phi": 0.3,
#         "bitstring": "0101",
#         "distribution": {
#             "00": 98,
#             "01": 309,
#             "10": 20,
#             "11": 764
#         }
#     }
# }