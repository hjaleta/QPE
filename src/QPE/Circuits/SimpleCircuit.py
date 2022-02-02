from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, circuit, execute
from qiskit.compiler import transpile, assemble
from math import pi
from QPE.Backend import GetBackend

def Miguelito():
    circuit = QuantumCircuit(1,1)
    circuit.initialize([1,0], 0)
    circuit.x(0)
    circuit.measure(0,0)
    return circuit

def Miguelito2():
    circuit = QuantumCircuit(1,1)
    circuit.initialize([1,0], 0)
    circuit.x(0)
    circuit.x(0)
    circuit.x(0)
    circuit.x(0)
    circuit.x(0)
    circuit.measure(0,0)
    return circuit

def Miguelito3():
    circuit = QuantumCircuit(2,2)
    circuit.initialize([1,0], 0)
    circuit.initialize([1,0], 1)
    circuit.x(1)
    circuit.measure(0,0)
    circuit.measure(1,1)
    return circuit

if __name__ == "__main__":
    b = GetBackend(service="local")
    o = transpile(Miguelito3())
    job = execute(o,b)
    print(job.result().get_counts())
    