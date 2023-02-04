#############################################################################
#An adaptation of [T-Fermion](https://github.com/PabloAMC/TFermion) to materials with nonorthogonal lattice
#Notations and equations follow the accompanying paper and [Su. et. al.](https://arxiv.org/pdf/2105.12767.pdf)
#############################################################################

import numpy as np
import itertools
from decimal import Decimal
import fire
from error_lambda import *
from utils import *
from init_state import *
from qubit_cost import *
from gate_cost import *
norm = np.linalg.norm

def first_quantization_qubitization(N : float, material_idx = 0 , pth = 0.95, n_parallel = 500):

    material = ['dis', 'limnfo', 'limnnio', 'limno'][material_idx]
    
    if material == 'dis':
        mspecs = {'vec_a': np.array([[5.02,0,0],[0,5.40,0],[0,0,6.26]]),
        'eta' : 156,
        'lambda_zeta': 156,
        'material_ortho_lattice': True,
        'n_p2cost' : {4: 2366, 5: 2867, 6: 3365, 9: 4859}
        }
    if material == 'limnfo':
        mspecs = {'vec_a': np.array([[12.48,0,0],[0,8.32,0],[0,0,8.32]]),
        'eta' : 863,
        'lambda_zeta': 863,
        'n_p2cost' : {4: 10906, 5: 13525, 6: 16147, 10: 26629},
        'material_ortho_lattice': True
        }
    if material == 'limnnio':
        mspecs = {'vec_a': np.array([[5.7081,0,0],[-4.2811,7.4151,0],[0,0,19.6317]]),
        'eta' : 968,
        'lambda_zeta': 968,
        'n_p2cost' : {4: 12171, 5: 15105, 6: 18045, 10: 29784},
        'material_ortho_lattice': False
        }
    if material == 'limno':
        mspecs = {'vec_a': np.array([[10.02,0,0],[0,17.32,0],[-1.6949, 0., 4.7995]]),
        'eta' : 808,
        'lambda_zeta': 808,
        'n_p2cost' : {4: 10248, 5: 12702, 6: 15156, 10: 24974},
        'material_ortho_lattice': False
        }
    
    print(locals())
    #mspecs
    vec_a, eta, lambda_zeta = mspecs['vec_a'], mspecs['eta'], mspecs['lambda_zeta']
    n_p2cost, material_ortho_lattice = mspecs['n_p2cost'], mspecs['material_ortho_lattice']
    
    #error_params
    n_errors_opt = 4
    error= 1.5e-3
    qpe_error_opt = 0.1
    e_opt = np.sqrt(error**2-error**2/(1+qpe_error_opt**2))/n_errors_opt

    epsilon_PEA = error/np.sqrt(1+qpe_error_opt**2)
    epsilon_M = e_opt
    epsilon_R = e_opt
    epsilon_S = e_opt
    epsilon_T = e_opt
    epsilon_B = e_opt 
    br = 8 #same as that used in PP
    optimized_parameters = [epsilon_PEA, epsilon_M, epsilon_R, epsilon_S, epsilon_T, br]
    weight_toffoli_cost = 1

    #basic params
    n_p = int(np.ceil(np.log2(N**(1/3) + 1)))
    N_small = N #needed for initial state preparation cost
    angs2bohr = 1.8897259886
    def compute_Omega(vecs):
        # print('Omega in Angs^3 ', np.abs(np.sum((np.cross(vecs[0],vecs[1])*vecs[2]))))
        return np.abs(np.sum((np.cross(vecs[0],vecs[1])*vecs[2]))) * angs2bohr**3 
    Omega = compute_Omega(vec_a)
    # print(f'Omega in bohr {Omega} in angs {Omega/ angs2bohr**3}')
    recip_bv = 2*np.pi/Omega * \
        np.array([np.cross(vec_a[i],vec_a[j]) for i,j in [(1,2),(2,0),(0,1)]]) * angs2bohr**2
    recip_bv_u, recip_bv_s, recip_bv_v = np.linalg.svd(recip_bv)
    lambda_min_B , lambda_max_B = np.min(recip_bv_s), np.max(recip_bv_s) 
    bmin = lambda_min_B
    # bmin2 = np.min([norm(bi) for bi in recip_bv])
    # assert abs(bmin2 - bmin) < 1e-6 #comment this out and you will see the assert goes thru --> The two nonorthogonal case studies have bmin = min ||b_i||

    #the partitioning of the lattice
    B_mus = {}
    for j in range(2, n_p+3):
        B_mus[j] = []
    if n_p <= 6: 
        for nu in itertools.product(range(-2**(n_p), 2**(n_p)+1), repeat = 3):
            nu = np.array(nu)
            if list(nu) != [0,0,0]:
                mu = int(np.floor(np.log2(np.max(abs(nu)))))+2
                B_mus[mu].append(nu)

    n_p, n_eta, n_eta_zeta, n_M, n_R, n_T, n_B, lambda_value, a_U = calculate_number_bits_parameters(optimized_parameters, N, n_p, eta, lambda_zeta, Omega, 
    recip_bv, B_mus, bmin, pth, epsilon_B, material_ortho_lattice)

    #number of QPE runs (overlap with g.s. not included)
    r = np.ceil(np.pi*lambda_value/(2*epsilon_PEA))
    
    epsilon_S_HF = 1e-2 #This parameter indicates the decrease in overlap with ground state due to imperfect rotations.
    epsilon_SS_HF = epsilon_S_HF / (2*eta*(N_small-eta))
    QPE_cost, beta, Prep, Sel = calculate_QPE_cost(n_p, n_eta, n_eta_zeta, n_M, n_R, n_T, lambda_value, r, br, eta, lambda_zeta, 
weight_toffoli_cost, epsilon_S, n_B, a_U, n_p2cost, n_parallel, material)
    HF_cost = calculate_HF_cost(epsilon_SS_HF, n_p, eta, N_small, weight_toffoli_cost, N)
    print('lambda ', lambda_value)
    # print('initial state Toffoli cost', "{:.2e}".format(Decimal(HF_cost)))
    print(f'Total Toffoli {"{:.2e}".format(Decimal(QPE_cost))} Prep {Prep} Sel {Sel}')
    qbit_cost, logical_clean_qubits, dirties_from_beta = qubit_cost(n_p, n_M, n_R, n_T, beta, lambda_value, epsilon_PEA, eta, lambda_zeta, n_B)
    # print('dirties from beta ', dirties_from_beta)
    print(f'Qubit cost {qbit_cost} clean {logical_clean_qubits}')
    

if __name__ == '__main__':
    fire.Fire(first_quantization_qubitization)