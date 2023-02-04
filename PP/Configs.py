import numpy as np

def build_Configs(order = 3, material_idx = 0, n_tof = 500, error=1.5e-3, pnuth=0.75):
    error = error #corresponding to 0.043 eV
    b_r = 8 #number of qubits used in the uniform superposition preparations, 8 is enough (very high probability of success)
    pnuth = pnuth #success threshold for amplitude amplification in loc and V
    orderofmag = order #N = 10**orderofmag
    material_name = ['dis','limnfo','limnnio','limno']  #dis: Li2FeSiO4, limnfo: Li0.5MnO2F, limnnio: Li0.75[Li0.17Mn0.58Ni0.25]O2, limno: Li0.5MnO3
    material_choice = material_name[material_idx]

    #baseparams
    #first component is the number of PW
    #second component is the number of qubits used in the AE algorithm for the same number of PW
    baseparams = {'dis': {3: (1e3,2366), 4: (1e4,2867), 5: (1e5,3365)}, 
    'limnfo': {3: (1e3,10906), 4: (1e4,13525), 5: (1e5,16147)},
    'limnnio': {3: (1e3,12171), 4: (1e4,15105), 5: (1e5,18045)},
    'limno': {3: (1e3,10248), 4: (1e4,12702), 5: (1e5,15156)}
    }
    N = baseparams[material_choice][orderofmag][0]
    n_dirty = baseparams[material_choice][orderofmag][1] #obtained by first running AE calculations
    kappa = 1 #we simply use a universal parallelization factor, instead of optimizing for each of qroms used in loc,V,NL
    n_parallel = n_tof #this is `n_tof` in the paper; quantum computer is capable of applying n_parallel many Toffolis simultaneously

    dilithium_iron_silicate = { #Li2FeSiO4
    'eta' : 100,
    'atoms_and_rep_uc' : {'Li': 4, 'O': 8, 'Si': 2, 'Fe': 2},  #how many atoms X are there in the uc (unit cell)
    'direct_basis_vectors': [[5.02,0,0],[0,5.40,0],[0,0,6.26]],
    'material_ortho_lattice': True, #if material has ortho lattice, the algorithm acts differently
    'Z_ions': {'Li': 1, 'O': 6, 'Si': 4, 'Fe': 8}, 
    # PP parameters, Bi_inv are the inverse values of the B_i's in the paper
    'PP_loc_params' : { 'Li': {'Z_ion':1, 'r_loc': 0.787553, 'Cs': [-1.892612, 0.286060]},
    'O': {'Z_ion':6, 'r_loc': 0.247621, 'Cs': [-16.580318, 2.395701]},
    'Si': {'Z_ion':4, 'r_loc': 0.440000, 'Cs': [-7.336103, 0.]},
    'Fe': {'Z_ion':8,'r_loc': 0.430000, 'Cs': [-6.654220, 0.]}},
    'PP_NL_params': {'Li': {'Z_ion':1 , 'rs': [0.666375, 1.079306, 0.], 'Bi_inv': [1.858811, -0.005895, 0.]},
    'O': {'Z_ion': 6, 'rs': [0.221786, 0.256829, 0.], 'Bi_inv': [18.266917, 0., 0.]},
    'Si': {'Z_ion': 4, 'rs': [0.422738, 0.484278, 0.], 'Bi_inv': [5.906928, 2.727013, 0.]},
    'Fe': {'Z_ion': 8, 'rs': [0.454482, 0.638903, 0.308732], 'Bi_inv': [3.016640, 1.499642, -9.145354]}},
    }

    Li_rich_manganese_oxifluoride = { #Li0.5MnO2F
    'eta' : 428,
    'atoms_and_rep_uc' : {'Li': 12, 'Mn': 16, 'F': 16, 'O': 32}, 
    'direct_basis_vectors': [[12.48,0,0],[0,8.32,0],[0,0,8.32]],
    'material_ortho_lattice': True,
    'Z_ions': {'Li': 1, 'Mn': 7, 'F': 7, 'O': 6},
    # PP parameters
    'PP_loc_params' : { 'Li': {'Z_ion':1, 'r_loc': 0.787553, 'Cs': [-1.892612, 0.286060]},
    'Mn': {'Z_ion':7, 'r_loc': 0.640000, 'Cs': [0, 0]},
    'F': {'Z_ion':7, 'r_loc': 0.218525, 'Cs': [-21.307361, 3.072869]},
    'O': {'Z_ion':6, 'r_loc': 0.247621, 'Cs': [-16.580318, 2.395701]}},
    'PP_NL_params': {'Li': {'Z_ion':1 , 'rs': [0.666375, 1.079306, 0.], 'Bi_inv': [1.858811, -0.005895, 0.]},
    'Mn': {'Z_ion': 7, 'rs': [0.481246, 0.669304, 0.327763], 'Bi_inv': [2.799031, 1.368776, -7.995418]},
    'F': {'Z_ion': 7, 'rs': [0.195567, 0.174268, 0.], 'Bi_inv': [23.584942, 0., 0.]},
    'O': {'Z_ion': 6, 'rs': [0.221786, 0.256829, 0.], 'Bi_inv': [18.266917, 0., 0.]}}
    }

    LLNMO = {#Li0.75[Li0.17Mn0.58Ni0.25]O2
    'eta' : 468,
    'atoms_and_rep_uc' : {'Li': 22, 'Mn': 14, 'Ni': 6, 'O': 48}, 
    'direct_basis_vectors': [[5.7081,0,0],[-4.2811,7.4151,0],[0,0,19.6317]],
    'material_ortho_lattice': False,
    'Z_ions': {'Li': 1, 'Mn': 7, 'Ni': 10, 'O': 6},
    # PP parameters
    'PP_loc_params' : { 'Li': {'Z_ion':1, 'r_loc': 0.787553, 'Cs': [-1.892612, 0.286060]},
    'Mn': {'Z_ion':7, 'r_loc': 0.640000, 'Cs': [0, 0]},
    'Ni': {'Z_ion':10, 'r_loc': 0.56, 'Cs': [0., 0.]},
    'O': {'Z_ion':6, 'r_loc': 0.247621, 'Cs': [-16.580318, 2.395701]}},
    'PP_NL_params': {'Li': {'Z_ion':1 , 'rs': [0.666375, 1.079306, 0.], 'Bi_inv': [1.858811, -0.005895, 0.]},
    'Mn': {'Z_ion': 7, 'rs': [0.481246, 0.669304, 0.327763], 'Bi_inv': [2.799031, 1.368776, -7.995418]},
    'Ni': {'Z_ion': 10, 'rs': [0.425399, 0.584081, 0.278113], 'Bi_inv': [3.619651, 1.742220, -11.608428]},
    'O': {'Z_ion': 6, 'rs': [0.221786, 0.256829, 0.], 'Bi_inv': [18.266917, 0., 0.]}}}

    Li_rich_manganese_oxide= {#Li0.5MnO3
    'eta' : 408,
    'atoms_and_rep_uc' : {'Li': 8, 'Mn': 16, 'O': 48}, 
    'direct_basis_vectors': [[10.02,0,0],[0,17.32,0],[-1.6949, 0., 4.7995]],
    'material_ortho_lattice': False,
    'Z_ions': {'Li': 1, 'Mn': 7, 'O': 6},
    # PP parameters
    'PP_loc_params' : { 'Li': {'Z_ion':1, 'r_loc': 0.787553, 'Cs': [-1.892612, 0.286060]},
    'Mn': {'Z_ion':7, 'r_loc': 0.640000, 'Cs': [0., 0.]},
    'O': {'Z_ion':6, 'r_loc': 0.247621, 'Cs': [-16.580318, 2.395701]}},
    'PP_NL_params': {'Li': {'Z_ion':1 , 'rs': [0.666375, 1.079306, 0.], 'Bi_inv': [1.858811, -0.005895, 0.]},
    'Mn': {'Z_ion': 7, 'rs': [0.481246, 0.669304, 0.327763], 'Bi_inv': [2.799031, 1.368776, -7.995418]},
    'O': {'Z_ion': 6, 'rs': [0.221786, 0.256829, 0.], 'Bi_inv': [18.266917, 0., 0.]}}
    }

    material_specs = {'dis':dilithium_iron_silicate, 'limnfo': Li_rich_manganese_oxifluoride,
                    'limnnio':LLNMO, 'limno':Li_rich_manganese_oxide}

    opt_error = True 
    #flag for some hand-designed optimal error config, not guaranteed to be the best of course, False means equal distribution of total error over all errors
    #set to False to see the impact of uniform error design on cost

    qrom_loc = True

    n_errors = 7

    #hand designed "optimal" error choice 
    qpe_error_opt = 0.1 #error_pha = total_error/sqrt(1+qpe_error_op**2)

    #notation: here error_NL is error_Psi in the paper, and error_k is error_NL in the paper
    #error_op is error_chi in the paper
    e = error/np.sqrt(n_errors**2+1)
    dict_errors_nopt = {'error_qpe': e, 'error_op': e, 
    'error_R': e, 
    'error_M_V': e, 'error_Mloc': e, 'error_NL': e,
    'error_k': e, 'error_b': e}

    e_opt  = np.sqrt(error**2-error**2/(1+qpe_error_opt**2))/n_errors
    dict_errors_opt = {'error_qpe': error/np.sqrt(1+qpe_error_opt**2),  'error_op': e_opt,
    'error_R': e_opt, 
    'error_M_V': e_opt, 'error_Mloc': e_opt, 'error_NL': e_opt,
    'error_k': e_opt, 'error_b': e_opt}

    dict_errors_choice = dict_errors_opt if opt_error else dict_errors_nopt


    Configs = { **material_specs[material_choice],
    'N': N,
    'kappa': kappa,
    'qrom_loc':qrom_loc,
    'n_parallel': n_parallel,
    'n_dirty': n_dirty,
    'opt_error': opt_error,
    **dict_errors_choice,
    'pnuth':pnuth,
    'b_r': b_r
    }
    return Configs