from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, circuit
from qiskit.compiler import transpile, assemble
from math import pi, sin, cos
from Scripts.Unitary import Unitary
import numpy as np

def KitaevCircuit(U, n_digits, states):
    dim = U.dim + 1
    N_states = len(states)
    N_class_regs = 2*N_states*n_digits
    circ = QuantumCircuit(dim, N_class_regs)
    circ.barrier()
    # circ.initialize([1,0], 0)
    target_qubits = list(range(1, dim))
    class_reg = 0
    
    for state in states:
        for n in range(n_digits):
            for rot in range(2):
                circ.initialize([1,0], 0)
                circ.initialize(state, target_qubits)
                circ.barrier()
                circ.h(0)
                if rot == 1:
                    circ.rz(pi/2,0)
                power = 2**(n_digits - 1 - n)
                u = U.get_gate(power, 1)
                circ.append(u, list(range(dim)))
                circ.h(0)
                
                circ.measure(0,class_reg)
                class_reg += 1
                if class_reg < N_class_regs: 
                    circ.barrier()

    return circ

if __name__ == "__main__":
    np.set_printoptions(precision=2)
    gods_phase = 0.3*2*pi
    # unitary_of_gods = Unitary([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, complex(cos(gods_phase),sin(gods_phase))]
    #     ])
    unitary_of_gods = Unitary([
        [1,0],
        [0,complex(cos(gods_phase),sin(gods_phase))]])
    
    k = KitaevCircuit(unitary_of_gods, 3, [[0,1]])
    d = k.draw("mpl")
    d.savefig("aplot.png")
    #backend = GetBackend("IBMQ")
    # circuit.draw()
    # mapped_circuit = transpile(circuit, backend=backend)
    # qobj = assemble(mapped_circuit, backend=backend, shots=1024)
    # job = backend.run(qobj)


    # pippiripappuro = 5