from utils import *
import numpy as np

"""Different cost parts: Givens, antisymmetry and QPE"""

def calculate_HF_cost(epsilon_SS_HF,n_p, eta, N_small, weight_toffoli_cost, N):

    # T gate cost for HF
    T_givens = c_pauli_rotation_synthesis(epsilon_SS_HF)# For C-Ry, lemma 5.4 Barenco
    HF_T_cost = eta*(N_small-eta)*T_givens
    
    # toffoli gate cost for HF
    aux = 2*(3*n_p-1)*eta # We can use m-1 Toffolis to perform a controlled not with m controls (and a few ancillas).
    swaps = (3*n_p)*(eta-1)
    Givens = 4*aux + 2*swaps # Compute and uncompute of the flag qubit, for p and q 
    HF_toffoli_cost = eta*(N_small-eta)*Givens + calculate_antisymmetrization_cost(eta,N)

    return  HF_toffoli_cost*weight_toffoli_cost
    # return HF_T_cost*weight_T_cost + HF_toffoli_cost*weight_toffoli_cost

def calculate_antisymmetrization_cost(eta,N):

    # Initial state antisymmetrization
    comparison_eta = compare_cost(np.ceil(np.log2(eta**2)))/4
    comparison_N = compare_cost(np.ceil(np.log2(N)))/4
    swaps_eta = np.ceil(np.log2(eta**2))
    swaps_N = np.ceil(np.log2(N))
    Step_2 = eta*np.ceil(np.log2(eta))*(np.ceil(np.log2(eta))-1)/4* (comparison_eta + swaps_eta)
    Step_4 = eta*np.ceil(np.log2(eta))*(np.ceil(np.log2(eta))-1)/4* (comparison_N + swaps_N)
    antisymmetrization_cost = Step_2*2 + Step_4 #the *2 is due to expected success rate

    return antisymmetrization_cost
