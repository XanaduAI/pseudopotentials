import numpy as  np
from scipy import integrate

def Pr(n):
    for j in range(1,int(n)):
        if n< j*(np.log(j)-np.log(np.log(2))-1)/np.log(2) + \
            (np.log(np.sqrt(2*np.pi)) - np.log(2))/np.log(2) + np.log(j)/(2*np.log(2)) - \
            1/((12*j+1)*np.log(2)):
            return j

#Pablo
def Ps(n, br): #eq 59 from https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040332

        theta = 2*np.pi/(2**br)*np.round((2**br)/(2*np.pi)*np.arcsin(np.sqrt(2**(np.ceil(np.log2(n)))/(4*n)))) #eq 60
        braket = (1+(2-(4*n)/(2**np.ceil(np.log2(n))))*(np.sin(theta))**2)**2 + (np.sin(2*theta))**2
        return n/(2**np.ceil(np.log2(n)))*braket

#Pablo
def pauli_rotation_synthesis(epsilon_SS):
    result = 10 + 4*np.ceil(np.log2(1/epsilon_SS))
    print(result)
    return result


#Pablo
def box_integral(f, N0):
    return integrate.nquad(f, [[1, N0],[1, N0]])[0]

def compute_clean_cost(g): #some costings are from the orthogonal case which is actually a tad bit larger than nonorthogonal
    #due to what happens in SEL_NL implementation , Also we assume simultaneous application of QROMs
    g.beta_loc = g.gate_cost_PREP_PP.beta_loc
    g.beta_V = g.gate_cost_PREP_PP.beta_V
    g.beta_k = g.gate_cost_PREP_PP.beta_k
    g.beta_NL = g.gate_cost_SEL_PP.beta_NL
    g.beta_NL_prime = g.gate_cost_SEL_PP.beta_NL_prime
    g.logn_a_max = g.gate_cost_PREP_PP.logn_a_max
    g.n_Ltype = g.gate_cost_PREP_PP.n_Ltype
    g.n_NL_prime = g.gate_cost_SEL_PP.n_NL_prime
    # the 35 in max([g.n_R+1, g.n_op, 35, g.n_b, g.n_k, g.n_NL]) belongs to the n_AA estimate for LIN
    clean_cost = (3*g.eta*g.n_p, np.ceil(np.log2(np.ceil(np.pi*g.lambda_val/(2*g.error_qpe)))),\
        max([g.n_R+1, g.n_op, 35, g.n_b, g.n_k, g.n_Mloc, g.n_NL_prime, g.n_NL]), 1 , 2, 4, 2*g.eta+5, 8, \
            2*g.eta, g.n_Ltype+g.logn_a_max, \
            g.n_Ltype+g.logn_a_max+5 , 4, 4, 2, 3*(g.n_p +1) , g.n_p, g.n_M_V,\
            3*g.n_p+2 , 2*g.n_p+1 , g.n_M_V, 1, 2, 3*(g.n_p+1), g.n_Ltype+ g.logn_a_max, 1,  \
            3+3+3+g.n_Ltype+9+4, 9, 2, 2*1) #2*1 for using n_AA 1 times in total
    clean_prep_temp_cost = max([5, 2*(g.n_Ltype + g.logn_a_max+1), g.n_k + \
        (g.n_Ltype+4), 4, g.n_Ltype+2, g.n_M_V+3*g.n_p, (g.n_Mloc+1)+(3*g.n_p+g.n_Ltype)+g.n_Mloc])
    clean_temp_H_cost = max([5*g.n_p+1, 5*g.n_R-4]) + max([clean_prep_temp_cost, 3*g.n_NL + \
        3*(g.n_p + g.n_Ltype+4), 3*g.n_p-1, 3*g.n_NL + 3*(g.n_p+g.n_Ltype+2) + 3])
    clean_temp_cost = max([clean_temp_H_cost, 2*g.n_eta + 9*g.n_p + g.n_Mloc + g.n_M_V + 35 + \
        2*(g.n_Ltype + g.logn_a_max)])
    return sum(clean_cost) + clean_temp_cost
            
def compute_dirty_cost(g):
    if g.material_ortho_lattice:
        return max([g.beta_k * g.n_k , g.beta_V * (g.n_M_V), g.beta_loc * (g.n_Mloc+1),\
            3*g.beta_NL*g.n_NL, 3*g.beta_NL_prime*g.n_NL])
    else:
        return max([g.beta_k * g.n_k , g.beta_V * (g.n_M_V), g.beta_loc * (g.n_Mloc+1),\
            2*g.beta_NL*g.n_NL, 2*g.beta_NL_prime*g.n_NL])