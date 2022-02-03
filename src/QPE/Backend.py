import json
from typing import Set
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from quantuminspire.credentials import enable_account as load_QI
from quantuminspire.qiskit import QI

def Login(service="IBMQ", token_path = "API_tokens.json", token = ""):


    if service not in ["IBMQ", "QI"]:
        raise KeyError('Choose service "IBMQ" or "QI"')

    if token:
        API_key = token
    else:
        try:
            with open(token_path, "r") as f:
                token_dict = json.load(f)
        except:
            raise FileNotFoundError(f"No json file {token_path} found")
        try:
            API_key = token_dict[service]
        except:
            raise KeyError(f"No token in {token_path} for service {service}")

    if service == "IBMQ":
        try:
            IBMQ.load_account()
        except:
            API_key = token_dict["IBMQ"]
            IBMQ.enable_account(API_key)
    elif service == "QI":
        API_key = token_dict["QI"]
        load_QI(API_key)
        QI.set_authentication()

def GetBackend(service="IBMQ", backend_name = "", flags:list = [], token_path = "API_tokens.json", token = ""):

    if service not in ["IBMQ", "QI", "local"]:
        raise KeyError('Choose service "IBMQ","QI" or "local"')

    if service in ["IBMQ", "QI"]:
        Login(service, token_path, token)
    
    sim = lambda x: x.configuration().simulator,
    Q5 = lambda x: x.configuration().n_qubits == 5 and not x.configuration().simulator

    if service == "IBMQ":
        provider = IBMQ.get_provider()
        if backend_name == "":
            if "5qubit" in flags:
                backend = least_busy(provider.backends(
                    filters= Q5
                    ))
            elif "simulator" in flags:
                backend = least_busy(provider.backends(
                    filters= sim
                    ))
        else:
            backend = provider.get_backend(backend_name)
    
    elif service == "QI":
        backend = QI.get_backend("Starmon-5")
        print(backend)
    
    elif service == "local":
        backend = Aer.get_backend("aer_simulator")

    print(service)
    return backend

filters = {
    "simulator": lambda x: x.configuration().simulator,
    "5qubit": lambda x: x.configuration().n_qubits == 5
                                   and not x.configuration().simulator
 }

if __name__ == "__main__":
    Login("IBMQ")
    #print(IBMQ.providers())
    backend = GetBackend("IBMQ")
    