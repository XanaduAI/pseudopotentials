import numpy as np

def qubit_cost(n_p,n_M,n_R,n_T,beta, lambda_val, error_qpe,eta, lambda_zeta, n_B):

    clean_cost = (3*eta*n_p, np.ceil(np.log2(np.ceil(np.pi*lambda_val/(2*error_qpe)))),  np.max([n_R+1,n_T,n_B]), 1, \
                  1, np.ceil(np.log2(eta+2*lambda_zeta))+3, 3+6,  2*np.ceil(np.log2(eta)) + 5, 3*(n_p+1), n_p, n_M, 3*n_p+2, 2*n_p+1,\
                  1, 2, n_M+1,2*n_p,6,1)    # +6 because of the register f previously 2 qubits were accounted for it
    clean_prep_temp_cost = max([5,n_M+3*n_p])
    clean_temp_H_cost = max([5*n_R-4,5*n_p+1]) + clean_prep_temp_cost
    clean_temp_cost = max([clean_temp_H_cost, np.ceil(np.log2(eta+2*lambda_zeta)) + 2*np.ceil(np.log2(eta))+ 6*n_p+n_M+16+3])
    logical_clean_qubits = sum(clean_cost) + clean_temp_cost 
    logical_qubits = max([logical_clean_qubits,beta*(n_M+1)])
    return logical_qubits, logical_clean_qubits, beta*(n_M+1)