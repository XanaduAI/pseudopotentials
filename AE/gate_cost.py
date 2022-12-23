import numpy as np


import numpy as np
from utils import *

def calculate_QPE_cost(n_p, n_eta, n_eta_zeta, n_M, n_R, n_T, lambda_value, r, br, eta, lambda_zeta, 
weight_toffoli_cost, epsilon_S, n_B, a_U, n_p2cost, n_parallel,material):


    # Section A
    ## weigthings between T and U+V, and U and V.
    weight_T_UV = n_T-3

    epsilon_SS_QPE = epsilon_S / (r*np.max([n_R+1, n_T])) # Denominator is r times size of gradient phase state: point 3 after eq C1 in https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040332

    eq_superp_T_UV = 3*n_eta_zeta + 2*br - 9
    ineq_test = n_eta_zeta - 1
    weight_U_V = eq_superp_T_UV +  ineq_test + 1

    prep_qubit_TUV = weight_T_UV + weight_U_V

    ## superposition between i and j
    bin_super = 3*n_eta + 2*br - 9
    equality_test = n_eta
    inv_equality_test = n_eta
    inv_bin_super = 3*n_eta + 2*br - 9

    prep_i_j = 2*bin_super + equality_test + inv_equality_test + 2*inv_bin_super

    ## success check
    success_check = 3

    # Section B: Qubitization of T

    ## Superposition w,r,s
    
    def binary_decomp_register_fgh(n_p, n_B):
        binary_decomp_register_fgh_cost = 2*(2*(2**(4+1)-1) + (n_B-3)*4 +\
             2**4 + (n_p-2))
        return binary_decomp_register_fgh_cost

    # sup_w = 3*2 + 2*br - 9 # = 3*n + 2*br - 9 with n = 2. br is suggested to be 8, no longer applies in the general lattice setting
    sup_w = binary_decomp_register_fgh(n_p, n_B)
    #T_sup_w = pauli_rotation_synthesis(epsilon_SS_QPE) + c_pauli_rotation_synthesis(epsilon_SS_QPE)
    sup_r = n_p - 2
    prep_wrs_T = 2*sup_r +sup_w # 2 for r and s
    prep_T_qubit = 5 #computing the flag qubit |.>_T that shows T was prepared. this is uncomputed using clifford and measurements.
    ## Sel T
    control_swap_i_j_ancilla = 2*2*(eta-2) #unary iteration for i and j, and for in and out
    swap_i_j_ancilla = 2*2*3*eta*n_p # 3 components, 2 for i and j, 2 for in and out. Also used in Sel_U_V
    cswap_p_q = control_swap_i_j_ancilla + swap_i_j_ancilla

    control_copy_w = 3*(n_p-1)
    copy_r = n_p - 1
    copy_s = n_p - 1
    control_phase = 1
    erasure = 0
    control_qubit_T = 1
    Sel_T = control_copy_w + copy_r + copy_s + control_phase + erasure + control_qubit_T

   
    def momentum_state_QROM(n_p, n_M, n_dirty, n_parallel, kappa):
        nqrom_bits = 3*n_p
        x = 2**nqrom_bits
        y = n_M+1 #one bit more due to binary decimal needed to compare to integer m
        
        beta_dirty = np.floor(n_dirty/y)
        beta_parallel = np.floor(n_parallel/kappa)
        
        if n_parallel == 1: 
            beta_gate = np.floor(np.sqrt(2*x/(3*y)))
        else:
            beta_gate = np.floor(2*x/(3*y/kappa) * np.log(2))
        
        if n_parallel == 1:
            beta = np.min([beta_dirty, beta_gate])
        else:
            beta = np.min([beta_dirty, beta_gate, beta_parallel])
            print('beta is determined by ',{0:'dirty',1:'gate',2:'parallel'}[np.argmin([beta_dirty, beta_gate, beta_parallel])])
        print('beta is ',beta, beta_dirty, beta_gate)
        if n_parallel == 1:
            momentum_state_cost_qrom = 2*np.ceil(x/beta) + 3*y*beta
        else:
            momentum_state_cost_qrom = 2*np.ceil(x/beta) + 3*np.ceil(y/kappa)*np.ceil(np.log2(beta))
            
        momentum_state_cost = 2*momentum_state_cost_qrom + y + 8*(n_p-1) + 6*n_p+2
        return momentum_state_cost, beta
    
    n_dirty = n_p2cost[n_p]
    #the prior derivation Prep_1_nu_and_inv_U is wrong since it uses quantum arithmetic but does not take into account the exact accuracy needed to compute G_nu^2
    Prep_1_nu_and_inv_U,beta  = momentum_state_QROM(n_p, n_M, n_dirty, n_parallel = n_parallel, kappa = 1)
    print('PREP cost for momentum state superposition and amplitude amplification' , Prep_1_nu_and_inv_U, a_U)
    # if a_U>1: 
    #   a_U = 1 #Lin's trick
    #   print('WARNING! LIN\'s trick application NOT CORRECT yet, use with caution, this is a lower bound for cost. applying Lin\'s trick for AA means a_U = ', a_U)
    #   Prep_1_nu_and_inv_U += 35-2 # this is for Lin's trick, which is used most of the times, so we just include it there by default.  
    
    
    QROM_Rl = lambda_zeta + Er(lambda_zeta)

    # Section D: Sel U and V

    swap_i_j_ancilla = 2*2*3*eta*n_p # 3 components, 2 for i and j, 2 for in and out  (Duplicated from Sel T)

    ## Controlled sum and substraction with change from signed integer to 2's complement
    signed2twos = 2*3*(n_p-2) # the 2 is for p and q, the 3 for their components
    addition = n_p+1
    controlled = n_p+1
    controlled_addition = 2*3*(addition + controlled)
    twos2signed = 2*3*(n_p) #Same as above, now with two extra qubits
    controlled_sum_substraction_nu = signed2twos + controlled_addition + twos2signed

    # No control-Z on |+> on Sel

    if n_R > n_p: # eq 97 # the product R_l \cdot k_\nu cancels the terms a_i
        U_phase = 3*(2*n_p*n_R-n_p*(n_p+1)-1)
    else:
        U_phase = 3*n_R*(n_R-1)

    #We could phase the T gates for each case, but instead it is faster to add the value into a phase gradient state: see https://quantum-journal.org/papers/q-2018-06-18-74/pdf/
    #U_phase_T_gates = (n_p+n_R+2)*pauli_rotation_synthesis(epsilon_SS_QPE) # arbitrary single rotations. The 2 comes from summing the three components
    
    # Total cost of Prepare and unprepare 
    Prep = 2*prep_qubit_TUV + prep_i_j + success_check + 2*prep_wrs_T + prep_T_qubit + (2*a_U+1)*Prep_1_nu_and_inv_U + QROM_Rl
    print('total PREP cost', Prep)
    # Total cost of Select
    Sel = cswap_p_q + Sel_T + controlled_sum_substraction_nu + U_phase
    print('total SEL cost', Sel)
    # Rotation in definition of Q
    Rot = n_eta_zeta + 2*n_eta + 6*n_p + n_M + 16 + 3 + 2 #3 is because 5-2=3, where 2 was already in the calculation of 16 for sup_w, and additional 2 for the n_AA

    # Final toffoli cost
    QPE_toffoli_cost = r*(Prep + Sel + Rot)

    # Final T cost: synthesis of the phase gradient state
    QPE_T_cost = np.max([n_R+1,n_T])*pauli_rotation_synthesis(epsilon_SS_QPE)

    # QPE_cost = QPE_toffoli_cost*weight_toffoli_cost + QPE_T_cost*weight_T_cost
    QPE_cost = QPE_toffoli_cost*weight_toffoli_cost 
  
    return QPE_cost,beta