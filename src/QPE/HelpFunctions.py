import numpy as np
from math import pi, sin, cos
import cmath
from typing import Union, List
from numpy.core.numeric import full

def im_to_tuple(im):
    return (im.real, im.imag)

def tuple_to_im(t):
    return complex(t[0],t[1])

def mat_im_to_tuple(M_i):
    dim = np.shape(M_i)
    M_t = np.zeros(dim).tolist()
    if len(dim) == 1:
        for i in range(dim[0]):
            M_t[i] = im_to_tuple(M_i[i])
    elif len(dim) == 2:
        for i in range(dim[0]):
            for j in range(dim[1]):
                M_t[i][j] = im_to_tuple(M_i[i][j])
    else:
        raise ValueError("the imput is not a vector or a matrix")
    return M_t
    # this is a list of list, not a numpy array

def mat_tuple_to_im(M_t):
    dim = np.shape(M_t)
    print(dim)
    M_i = np.zeros(dim).tolist()
    if len(dim) == 2:
        for i in range(dim[0]):
            M_i[i] = tuple_to_im(M_t[i])
    elif len(dim) == 3:
        for i in range(dim[0]):
            for j in range(dim[1]):
                M_i[i][j] = tuple_to_im(M_t[i][j])
    else:
        raise ValueError("the imput is not a vector or a matrix")
    return np.array(M_i, dtype=np.complex128)



def float_to_bin(x, n_digits:int):
    """ 
    Convert a number x in range [0,1] to a binary string truncated to length n_digits

    arguments:
        x: float 
        n_digits: integer
    
    return: 
        x_bin: string
        The decimal representation of digits AFTER '0.'
        Ex:
            Input 0.75 has binary representation 0.11 
            Then this function would return '11'
    """
    if x < 0 or x >= 1:
        raise ValueError("x must be in interval [0,1)")
    x_round = round(x * 2**n_digits)
    # print(x_round)
    # print(2**n_digits)
    if x_round == 2**n_digits:
        x_round = 0
    x_raw = bin(x_round)
    x_bin = x_raw[2:].zfill(n_digits)
    return x_bin

def bin_to_float(b:Union[str, list]):
    if isinstance(b, list):
        s = ""
        # print(b)
        for digit in b:
            if len(digit) != 1:
                raise ValueError("Invalid input")
            s += digit
    if b[:2] == "0.":
        b = b[2:]
    f = 0
    for d_i, digit in enumerate(b, 1):
        if digit not in "01":
            # print(digit)
            raise ValueError("Invalid input")
        if digit == "1":
            f += 2**-d_i

    return f

def phase_to_exp(phi, radians = False):
    """
    Returns the complex exponential e^(i*phi*2*pi)
    If radians == True it returns e^(i*phi) 
    """
    phi = np.array(phi)
    if radians:
        phase = phi*1j
    else:
        phase = phi*2*pi*1j
    return np.exp(phase)

def count_conversion(counts):
    """Converts a dictionary of combined measurement counts to a list with the number of individual counts per classical channel
    Args:
        counts: dict
            The dictionary version of the qiskit.result.counts object
    
    returns:
        ones: 1D numpy array
            The count of the individual bits, in particular the number of ones
            first element = number of times a 1 was measured in the first register
    
    Ex:
        input: counts = {'00': 5, "01": 10, "10": 100, "11": 500}
        return: [600, 510]
    """
    if not isinstance(counts, dict):
        try:
            counts = dict(counts)
        except:
            raise TypeError("Input counts is not dict or convertable to dict")
    
    n_bits = len(list(counts.keys())[0])
    ones = np.zeros(n_bits)
    for bit_string in counts:
        result_count = counts[bit_string]
        for d_i, digit in enumerate(bit_string):
            if digit == "1":
                ones[d_i] += result_count
    return ones

def get_sub_bitstring_counter(counts, n_splits):
    if not isinstance(counts, dict):
        try:
            counts = dict(counts)
        except:
            raise TypeError("Input counts is not dict or convertable to dict")

    n_bits = len(list(counts.keys())[0])
    if n_bits%n_splits != 0:
        raise ValueError("Length of bitstring and n_splits not compatible")
    sub_bitstring_len = n_bits//n_splits
    sub_bitstring_dicts = []
    for n in range(n_splits):
        d = {}
        b_start = n*sub_bitstring_len
        b_end = (n+1)*sub_bitstring_len
        for full_bitstring in counts:
            temp_count = counts[full_bitstring]
            sub_bitstring = full_bitstring[b_start:b_end]
            if sub_bitstring in d.keys():
                d[sub_bitstring] += temp_count
            else:
                d[sub_bitstring] = temp_count
        sub_bitstring_dicts.append(d)
    
    return sub_bitstring_dicts

def calculate_bitstring_distribution(ones_list:list, bitstring_length:int, n_shots:int):
    
    # print()
    assert 2**bitstring_length -1 == len(ones_list)
    # indices = [2**j - 1 for j in range(bitstring_length)]
    # most_significant_digits = ones_list[-index:]
    bitstring_dict = {"":1}
    i_bit = 0
    while i_bit < bitstring_length:
        index = 2**i_bit
        ones = ones_list[-index:]
        new_bitstring_dict = {}
        for key in bitstring_dict:
            p_key = bitstring_dict[key]
            # print(p_key)
            key_index = 0 if key == "" else int(key, 2) 
            p_one = p_key * ones[key_index]/n_shots
            p_zero = p_key*(1 - ones[key_index]/n_shots)
            # print(p_one, p_zero)
            new_bitstring_dict["0"+key] = p_zero
            new_bitstring_dict["1" + key] = p_one
        bitstring_dict = new_bitstring_dict
        i_bit += 1
    return bitstring_dict

def normalize_counts(counts):
    norm_count = []
    for count in range(counts):
        if not isinstance(counts, dict):
            try:
                count = dict(count)
            except:
                raise TypeError("Input counts is not dict or convertable to dict")
    
        norm_count.append({key:count[key]/2 for key in count})

    return norm_count

if __name__ == "__main__":
    np.set_printoptions(precision=2)

    ones_list = [4,6,2, 34, 32, 9, 27]
    n_shots = 50
    b_len = 3

    k = calculate_bitstring_distribution(ones_list, b_len, n_shots)

    j = sum([k[l] for l in k])

    print(k)
    print(j)
    # k = np.arange(0,1,0.1)
    # # print(k)
    # l = phase_to_exp(k)
    # print(l)
    # test_dict = {'0000': 5, "0001": 10, "1101": 100, "1111": 500}
    # c = get_sub_bitstring_counter(test_dict, 2)
    # print(c)

    # n = 0.3
    # l = float_to_bin(n, 8)
    # print(l)
    # k = bin_to_float(l)
    # print(k)
    
    # h = "01001100"
    # print(bin_to_float(h))


    # M = np.array([[1+1j,0+0j],[1+1j,2+1j]], dtype=np.complex128)
    # M = np.zeros(2)
    # dim = np.shape(M)
    # print(len(dim))
    # M_new = mat_im_to_tuple(M)
    # print(M_new)
    # M_2 = mat_tuple_to_im(M_new)
    # print(M_2)