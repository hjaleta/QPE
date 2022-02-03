import json
from QPE.HelpFunctions import mat_im_to_tuple, mat_tuple_to_im
from QPE.Unitary import Unitary
from QPE.QFT import QFT
from QPE.Kitaev import Kitaev
from QPE.Iterative import Iterative

def unpack_PE_from_json(path):

    with open(path, "r") as json_object:
        d = json.load(json_object)

    estimator = d["Estimator"]
    
    tuple_matrix = d["Unitary"]["data"]
    print(tuple_matrix)
    print(type(tuple_matrix))
    im_matrix = mat_tuple_to_im(tuple_matrix)
    U = Unitary(im_matrix)

    n_digits = d["Bits of precision"]
    input_states = [mat_tuple_to_im(d["Input state"])]
    n_shots = d["shots"]
    backend_params = {"service": d["Backend"]["service"]}
    if backend_params["service"] in ["IBMQ", "QI"]:
        backend_params["backend_name"] = d["Backend"]["name"]
    method_specific_params = d["method_specific_params"]
    re_initialize = d["re_initialize"]
    
    if estimator == "QFT":
        PE = QFT
    elif estimator == "Kitaev":
        PE = Kitaev
    elif estimator == "Iterative":
        PE = Iterative
    PE_object = PE(U, n_digits, n_shots=n_shots,
            backend_params=backend_params, input_states=input_states,
            method_specific_params=method_specific_params, re_initialize=re_initialize)
    

    PE_object.estimated_phases = [d["estimated phase"]["phi"]]
    PE_object.estimated_bits = [d["estimated phase"]["bitstring"]]
    PE_object.bitstring_distributions = [d["estimated phase"]["distribution"]]
    
    return PE_object

if __name__ == "__main__":
    p = "example_experiment_hardware.json"
    PE = unpack_PE_from_json(p)