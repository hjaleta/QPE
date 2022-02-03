import numpy as np
from QPE.HelpFunctions import float_to_bin, mat_im_to_tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from scipy.stats import unitary_group
from math import pi

# np.array()

class Unitary():
    def __init__(self, operator = [], random = False, random_state=None, dim = 1, name = ""):
        self.name = name
        self.operator = self.get_operator(operator, random, random_state, dim)
        self.dim = self.get_dim()
        self.eigen = self.get_eigen()
        self.vectors = self.get_vectors()
        self.phis  = self.get_phis()
    
    def get_operator(self, operator, random, random_state, dim):

        if random == True:
            self.dim = dim
            operator = self.gen_random(random_state)
        else:
            try:
                operator = Operator(operator)
            except:
                raise(ValueError('insert a list, an operator or a circuit'))
        return operator

    def gen_random(self, random_state):
        return Operator(unitary_group.rvs(2**self.dim, random_state=random_state))

    def get_dim(self):
        return int(np.log2(self.operator.dim[0]))

    def get_gate(self, power=1, ctrl=0):  

        changed_op = self.operator.power(power)

        circ_ctr = QuantumCircuit(self.dim)
        circ_ctr.append(changed_op,list(range(self.dim)))
        lab = r'$U^{'+str(power)+r'}$' if power != 1 else 'U'

        if ctrl != 0:
            changed_op = circ_ctr.to_gate(label = lab).control(ctrl)
        else:
            changed_op = circ_ctr.to_gate(label = lab)
    
        return changed_op
        
        # Return gate that can be appended to circuit U^power with ctrl control qubits
        # ex.
        # circ.append(U, [0,1,2])
        # if ctrl = 1 then the control qubit is the fist one
        # if ctrl = 2 then the control qubits is the fist two
    
    def get_eigen(self):
        # self.operator.data
        eigval, eigvect = np.linalg.eig(self.operator.data)
        # print(eigval, eigvect)
        eig = np.concatenate((np.array(eigval).reshape(len(eigval),1), eigvect), axis=1)
        eigenvectors = []
        for i in range(len(eigval)):
            value = eig[i,0]
            vector = eig[i,1:]
            el = Eigenvector(vector, value, i)
            eigenvectors.append(el)

        return eigenvectors
        # Return a matrix 
        # Each raw is [eigenvalue, eigenvector[0], eigenvector[1],...eigenvector[2**dim]]
    
    def get_vectors(self):
        vectors = []
        for e in self.eigen:
            vectors.append(e.vector)
        return vectors

    def get_phis(self):
        phis = []
        for e in self.eigen:
            phis.append(e.phi)
        return phis
    
    def get_phi_bitstrings(self, n_digits):
        b_list = []
        for phi in self.phis:
            b = float_to_bin(phi, n_digits)
            b_list.append(b)
        return b_list
    
    def get_dict(self, n_digits):
        d = {}
        d["dimension"] = self.dim
        d["data"] = mat_im_to_tuple(self.operator.data)
        d["eigenvectors"] = mat_im_to_tuple(self.vectors)
        d["phis"] = mat_im_to_tuple(self.phis)
        d["phi_bitstrings"] = self.get_phi_bitstrings(n_digits)
        return d


class Eigenvector():
    def __init__(self, vector, value, index):
        self.vector = vector
        self.value = value 
        self.index = index
        self.phi = self.get_phi()    
    
    def get_phi(self):
        phi = np.angle(self.value)/(2*pi)
        if phi < 0:
            phi += 1
        return phi
        #Return the phase , i.e. a number between 0 and 1
        # The numpy.angle function could probably be useful
    
    def get_phi_bitstring(self, n_digits):
        bitstring = float_to_bin(self.phi, n_digits)
        return bitstring


if __name__ == "__main__":
    u = Unitary(random=True)
    #print(u.operator.data)
    d = u.get_dict(3)
    print(d)
    # k = u.eigen[0].get_phi_bitstring(n_digits = 10)
    # print(k)
    # phi = u.eigen[0].get_phi()
    # print(phi)
