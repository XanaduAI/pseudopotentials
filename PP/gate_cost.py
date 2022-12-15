import numpy as np
from utils import compute_dirty_cost, compute_clean_cost
from lambda_X import lambda_X_HGH
from Configs import Configs
from decimal import Decimal


class gate_cost_PREP_PP:
    def __init__(self, **kwargs):
        """ 
        expecting the following parameters for all methods to work:
        eta, L, n_p, b_r, n_M_V, n_b, n_Mloc, n_NL, AA (# of amplitude amplification)
        """
        for param, val in kwargs.items():
            setattr(self, param, val)
            if param == 'eta':
                self.n_eta = np.ceil(np.log2(self.eta))
        self.list_atoms, self.atoms_rep_uc = list(self.atoms_and_rep_uc.keys()), np.array(list(self.atoms_and_rep_uc.values()))
        self.natoms_type = len(self.list_atoms)
        self.n_Ltype = np.ceil(np.log2(self.natoms_type))
        self.atoms_rep = self.atoms_rep_uc
        self.n_a_max = max([self.atoms_rep[idx] for idx, atom in enumerate(self.list_atoms)])
        self.logn_a_max = np.ceil(np.log2(self.n_a_max))
        if self.qrom_loc: 
                    self.momentum_state_loc = self.momentum_state_loc_qrom
        
    def momentum_state_loc_qrom(self):
        n_qrom_bits = 3*self.n_p+self.n_Ltype
        x = 2**(n_qrom_bits + 1)
        y = self.n_Mloc * (n_qrom_bits-1) + (self.n_Mloc+1)  #the very last one has to output sign k_i(G_nu)
        
        beta_loc_dirty = np.floor(self.n_dirty/(self.n_Mloc+1))
        
        if self.n_parallel == 1: 
            beta_loc_gate = np.floor(np.sqrt(2*x/(3*(self.n_Mloc+1))))
        else:
            beta_loc_gate = np.floor(2*x/(3*(self.n_Mloc+1)/self.kappa) * np.log(2))
            
        beta_loc_parallel = np.floor(self.n_parallel/self.kappa)
        
        if self.n_parallel == 1:
            self.beta_loc = np.min([beta_loc_dirty,beta_loc_gate])
        else:
            self.beta_loc = np.min([beta_loc_dirty,beta_loc_gate,beta_loc_parallel])
        
        
        if self.n_parallel == 1:
            self.momentum_state_qrom_loc_cost = 2*np.ceil(x/self.beta_loc) + \
            3*self.beta_loc *(self.n_Mloc)*(3*self.n_p-1)+\
                3*self.beta_loc * (self.n_Mloc+1) + 6*self.n_p
        else:
            self.momentum_state_qrom_loc_cost = 2*np.ceil(x/self.beta_loc) + \
            3*np.ceil(np.log2(self.beta_loc)) * np.ceil((self.n_Mloc)/self.kappa)*(3*self.n_p-1) +\
                3*np.ceil(np.log2(self.beta_loc)) * np.ceil((self.n_Mloc+1)/self.kappa) + 6*self.n_p
        
        self.momentum_state_loc_cost = 2*(self.momentum_state_qrom_loc_cost) + (self.n_Mloc-3)*n_qrom_bits
        return self.momentum_state_loc_cost 
    
    def prep_sel_qubits(self):
        self.prep_sel_qubits_cost = 2*(2* 2**(2+1)+2*(self.n_op-3)*2)
        return self.prep_sel_qubits_cost
    
    def prep_R_nuclei_coords(self):
        self.prep_R_nuclei_coords_cost = 2*2*(2**(self.n_Ltype+np.ceil(np.log2(self.n_a_max))+1))
        return self.prep_R_nuclei_coords_cost

    def prep_uniform_nuclei_type_loc(self):
        self.v2natoms_type =  np.ceil(np.log2((self.natoms_type & (~(self.natoms_type - 1)))))
        self.prep_uniform_nuclei_type_loc_cost = 2*(3* self.n_Ltype -\
             3*self.v2natoms_type+2*self.b_r-9)
        return self.prep_uniform_nuclei_type_loc_cost

    def prep_uniform_nuclei_a_locV(self):
        self.v2n_a_max =  np.ceil(np.log2((self.n_a_max & (~(self.n_a_max - 1)))))
        self.prep_uniform_nuclei_a_locV_cost = 2*2*(3* np.ceil(np.log2(self.n_a_max)) -\
             3*self.v2n_a_max + 2*self.b_r-9 + 2 * 2**self.n_Ltype)
        return self.prep_uniform_nuclei_a_locV_cost

    def prep_nondiagonal_V(self):
        self.prep_nondiagonal_V_cost =  14*self.n_eta + 8*self.b_r - 36
        return self.prep_nondiagonal_V_cost

    def binary_decomp_register_fgh(self):
        self.binary_decomp_register_fgh_cost = 2*(2*(2**(4+1)-1) + (self.n_b-3)*4 +\
             2**4 + (self.n_p-2))
        return self.binary_decomp_register_fgh_cost

    def prep_op_registers(self):
        self.prep_op_registers_cost = 6+3+3+self.n_Ltype+9
        self.prep_NL_c_register_cost = 4
        self.prep_op_registers_cost += self.prep_NL_c_register_cost
        return self.prep_op_registers_cost

    def binary_decomp_indices(self):
        o_idx = 13 
        n_qrom_bits = np.ceil(np.log2(o_idx))+self.n_Ltype  
        x = 2**(n_qrom_bits+1)-1 
        y = self.n_k * n_qrom_bits
        beta_k_dirty = np.floor(self.n_dirty/y)
        beta_k_parallel = np.floor(self.n_parallel/self.kappa)
        beta_k_gate = np.floor(2*x/(3*y/self.kappa) * np.log(2))
        self.beta_k = np.min([beta_k_dirty, beta_k_gate, beta_k_parallel])
        if self.beta_k == 0: 
            self.beta_k=1
            print('lambda = 1')
        qrom_cost = 2*(2*np.ceil(x/self.beta_k) + 3*np.ceil(self.n_k/self.kappa)*\
                np.ceil(np.log2(self.beta_k))*n_qrom_bits + 2*n_qrom_bits)+\
                (self.n_k-3)*n_qrom_bits
        self.binary_QROM_decomp_indices_cost = 2*qrom_cost + 2*2**(self.n_Ltype+2) + 12 
        
        return self.binary_QROM_decomp_indices_cost

    def qrom_prep(self):
        self.qrom_prep_cost = 2*(self.natoms+self.erase(self.natoms))
        return self.qrom_prep_cost

    def momentum_state_V(self):
        nqrom_bits = 3*self.n_p
        x = 2**nqrom_bits
        y = self.n_M_V+1 #one bit more due to binary decimal needed to compare to integer m
        beta_V_dirty = np.floor(self.n_dirty/y)
        beta_V_parallel = np.floor(self.n_parallel/self.kappa)
        
        
        
        if self.n_parallel == 1: 
            beta_V_gate = np.floor(np.sqrt(2*x/(3*y)))
        else:
            beta_V_gate = np.floor(2*x/(3*y/self.kappa) * np.log(2))
            
            
        if self.n_parallel == 1:
            self.beta_V =np.min([beta_V_dirty, beta_V_gate])
        else:
            self.beta_V = np.min([beta_V_dirty, beta_V_gate, beta_V_parallel])
        
        
        if self.n_parallel == 1:
            self.momentum_state_V_cost_qrom = 2*np.ceil(x/self.beta_V) + 3*y*self.beta_V
        else:
            self.momentum_state_V_cost_qrom = 2*np.ceil(x/self.beta_V) + 3*np.ceil(y/self.kappa)*np.ceil(np.log2(self.beta_V))
        
        self.momentum_state_V_cost = 2*self.momentum_state_V_cost_qrom + y + 8*(self.n_p-1) + 6*self.n_p+2
        # self.momentum_state_V_cost += 35-2 #n_AA estimate for LIN
        return self.momentum_state_V_cost

    def momentum_state_loc(self):
        n_qrom_bits = 3*self.n_p+self.n_Ltype
        x = 2**n_qrom_bits
        
        beta_loc_dirty = np.floor(self.n_dirty/(self.n_Mloc+2))
        beta_loc_gate = np.floor(2*x/(3*(self.n_Mloc+2)/self.kappa) * np.log(2))
        beta_loc_parallel = np.floor(self.n_parallel/self.kappa)
        self.beta_loc = np.min([beta_loc_dirty,beta_loc_gate,beta_loc_parallel])
        
        # print('NUMBER OF DIRTIES REQUIRED FOR PREP_LOC', self.n_dirty)
        
        self.ineq_qrom_cost = 2*np.ceil(x/self.beta_loc) + \
            3*np.ceil(np.log2(self.beta_loc)) * np.ceil((self.n_Mloc+2)/self.kappa)
        # print('PARALLELIZATION REQUIRED FOR PREP_LOC', self.beta_loc * self.kappa)
        
        self.momentum_state_loc_cost = 2*self.ineq_qrom_cost + 8*(self.n_p-1) + 6*self.n_p + 2 + (self.n_Mloc+1)
        self.momentum_state_loc_cost += 35-2 #n_AA estimate for LIN
        return self.momentum_state_loc_cost   

    def prep_T_QROM(self):
        nqrom_bits = 4
        x = 2**(nqrom_bits+1) -1 
        y = self.n_b * nqrom_bits
        beta_T_qrom  = np.ceil(np.sqrt(x/y))
        self.prep_T_QROM = 2 * 2 * (np.ceil(x/beta_T_qrom)+ beta_T_qrom*y + nqrom_bits * (self.n_b-3))
        return self.prep_T_QROM
    
    def cost(self):
        self.cost = self.prep_nondiagonal_V() + self.binary_decomp_indices() + self.prep_T_QROM() + \
            self.prep_uniform_nuclei_type_loc() + self.prep_uniform_nuclei_a_locV() + \
            self.prep_R_nuclei_coords() + self.binary_decomp_register_fgh() + \
            self.prep_op_registers() + self.AA_V*self.momentum_state_V() + \
            self.AA_loc*self.momentum_state_loc()
        return self.cost

    #Pablo
    def erase(self, x):
        logx = np.log2(x)
        fres = 2**(np.floor(logx/2)) + np.ceil(2**(-np.floor(logx/2))*x)
        cres = 2**(np.ceil(logx/2)) + np.ceil(2**(-np.ceil(logx/2))*x)
        return min(fres, cres)



class gate_cost_SEL_PP:
    def __init__(self, **kwargs):
        """ 
        expecting the following parameters for all methods to work:
        eta, n_p, n_R, n_NL, n_b
        """
        for param, val in kwargs.items():
            setattr(self, param, val)
        self.n_Ltype  =  np.ceil(np.log2(len(list(self.atoms_and_rep_uc.keys()))))

    def cswap(self):
        self.cswap_cost = 12*self.eta*self.n_p + 4*self.eta - 8
        return self.cswap_cost
    
    def selcosts_binary_QROM_decomp_cost(self):
        self.selcosts_binary_QROM_decomp_cost = 5*(self.n_p - 1) + 2 + \
            2*self.selcosts_QROM() 
        return self.selcosts_binary_QROM_decomp_cost

    def selcosts_QROM(self):
        ortho_additional = self.n_Ltype + 4

        if self.material_ortho_lattice:
            n_qrom_bits = self.n_p + ortho_additional
        else:
            n_qrom_bits = 2*self.n_p + ortho_additional


        x = 2**(n_qrom_bits +1)-2**(ortho_additional) 
        
        if self.material_ortho_lattice:
            y = self.n_NL * self.n_p
        else:
            y = self.n_NL * 2* self.n_p
        
        if self.material_ortho_lattice:
            beta_NL_dirty = np.floor(self.n_dirty/(3*self.n_NL))
            beta_NL_gate = np.floor(2*x/(3*(y / self.kappa)) * np.log(2))
            beta_NL_parallel = np.floor(self.n_parallel/(3*self.kappa))
            self.beta_NL = np.min([beta_NL_dirty,
                beta_NL_gate, beta_NL_parallel])
        else:
            beta_NL_dirty = np.floor(self.n_dirty/(2*self.n_NL))
            beta_NL_gate = np.floor(2*x/(3*(y / self.kappa)) * np.log(2))
            beta_NL_parallel = np.floor(self.n_parallel/(2*self.kappa))
            self.beta_NL = np.min([beta_NL_dirty,
                beta_NL_gate, beta_NL_parallel])
        
        if self.n_parallel == 1:
            beta_NL_gate = np.floor(np.sqrt(2*x/(3*y)))
            self.beta_NL = np.min([beta_NL_dirty, beta_NL_gate])
        
        print('NUMBER OF DIRTIES REQUIRED FOR SEL_NL', self.n_dirty)
        
        if self.material_ortho_lattice:
            if self.n_parallel == 1:
                self.selcosts_QROM_cost = 2*(2* np.ceil(x/self.beta_NL) + \
                        3* self.beta_NL * \
                            self.n_NL*self.n_p + \
                        2*self.n_p)+(self.n_NL-3)*y/self.n_NL + (3*self.n_p-1)
                self.selcosts_QROM_cost *= 3
            else:
                self.selcosts_QROM_cost = 2*(2* np.ceil(x/self.beta_NL) + \
                        3* np.ceil(np.log2(self.beta_NL)) * \
                            np.ceil(self.n_NL / self.kappa)*self.n_p + \
                        2*self.n_p)+(self.n_NL-3)*y/self.n_NL + (3*self.n_p-1)
        else:
            if self.n_parallel == 1:
                self.selcosts_QROM_cost = 2*(2* np.ceil(x/self.beta_NL) + \
                        3* self.beta_NL * \
                            self.n_NL*2*self.n_p + \
                        2*2*self.n_p)+(self.n_NL-3)*y/self.n_NL + (3*self.n_p-1)
                
                n_qrom_bits_n_p = self.n_p + ortho_additional
                x_n_p = 2**(n_qrom_bits_n_p +1)-2**(ortho_additional)
                y_n_p = self.n_NL * self.n_p
                
                self.selcosts_QROM_cost += 2*(2* np.ceil(x_n_p/self.beta_NL) + \
                        3* self.beta_NL * \
                            self.n_NL*self.n_p + \
                        2*self.n_p)+(self.n_NL-3)*y_n_p/self.n_NL + (3*self.n_p-1)
            else:     
                self.selcosts_QROM_cost = 2*(2* np.ceil(x/self.beta_NL) + \
                        3* np.ceil(np.log2(self.beta_NL)) * \
                            np.ceil(self.n_NL / self.kappa)*2*self.n_p + \
                        2*2*self.n_p)+(self.n_NL-3)*y/self.n_NL + (3*self.n_p-1)





        print('the sel costs qrom',self.selcosts_QROM_cost)
        ########
        ortho_additional_20 = self.n_Ltype + 2

        if self.material_ortho_lattice:
            n_qrom_bits_20 = self.n_p + ortho_additional_20
            x = 2**(n_qrom_bits_20 +1)-2**(ortho_additional_20) 
            y = self.n_NL * self.n_p
        else:
            n_qrom_bits_20 = 2*self.n_p + ortho_additional_20
            x = 2**(n_qrom_bits_20 +1)-2**(ortho_additional_20) 
            y = self.n_NL * 2*self.n_p


        self.n_NL_prime = 50 #default high value set by us

        if self.material_ortho_lattice:
            beta_NL_prime_dirty = np.floor(self.n_dirty/(3*self.n_NL))
            beta_NL_prime_gate = np.floor(2*x/(3*(y / self.kappa)) * np.log(2))
            beta_NL_prime_parallel = np.floor(self.n_parallel/(3*self.kappa))
            self.beta_NL_prime = np.min([beta_NL_prime_dirty,
                beta_NL_prime_gate, beta_NL_prime_parallel])
        else:
            beta_NL_prime_dirty = np.floor(self.n_dirty/(2*self.n_NL))
            beta_NL_prime_gate = np.floor(2*x/(3*(y / self.kappa)) * np.log(2))
            beta_NL_prime_parallel = np.floor(self.n_parallel/(2*self.kappa))
            self.beta_NL_prime = np.min([beta_NL_prime_dirty,
                beta_NL_prime_gate, beta_NL_prime_parallel])

        if self.n_parallel == 1:
            beta_NL_prime_gate = np.floor(np.sqrt(2*x/(3*y)))
            self.beta_NL_prime = np.min([beta_NL_prime_dirty,
                beta_NL_prime_gate])

        self.selcosts_QROM_i_cost = 2*(2**(3+1)-1) + 3*(self.n_NL_prime-3)
        print('NUMBER OF DIRTIES REQUIRED FOR SEL_NL', self.n_dirty)

        if self.material_ortho_lattice:
            if self.n_parallel == 1:
                self.selcosts_QROM_o_cost = 2*(2* np.ceil(x/self.beta_NL_prime) + \
                        3* self.beta_NL_prime * \
                            self.n_NL *self.n_p + \
                        2*self.n_p)+(self.n_NL-3)*y/self.n_NL
                self.selcosts_QROM_o_cost *= 3
            else:
                self.selcosts_QROM_o_cost = 2*(2* np.ceil(x/self.beta_NL_prime) + \
                        3* np.ceil(np.log2(self.beta_NL_prime)) * \
                            np.ceil(self.n_NL / self.kappa)*self.n_p + \
                        2*self.n_p)+(self.n_NL-3)*y/self.n_NL
        else:
            if self.n_parallel == 1:
                self.selcosts_QROM_o_cost = 2*(2* np.ceil(x/self.beta_NL_prime) + \
                        3* self.beta_NL_prime * \
                            self.n_NL *2*self.n_p + \
                        2*2*self.n_p)+(self.n_NL-3)*y/self.n_NL
                
                n_qrom_bits_n_p = self.n_p + ortho_additional_20
                x_n_p = 2**(n_qrom_bits_n_p +1)-2**(ortho_additional_20)
                y_n_p = self.n_NL * self.n_p
                
                self.selcosts_QROM_o_cost += 2*(2* np.ceil(x/self.beta_NL_prime) + \
                        3* self.beta_NL_prime * \
                            self.n_NL *self.n_p + \
                        2*self.n_p)+(self.n_NL-3)*y/self.n_NL
            else:
                self.selcosts_QROM_o_cost = 2*(2* np.ceil(x/self.beta_NL_prime) + \
                        3* np.ceil(np.log2(self.beta_NL_prime)) * \
                            np.ceil(self.n_NL / self.kappa)*2*self.n_p + \
                        2*2*self.n_p)+(self.n_NL-3)*y/self.n_NL

        if self.material_ortho_lattice:
            self.selcosts_QROM_20_cost = 5*(self.selcosts_QROM_i_cost + \
                self.selcosts_QROM_o_cost + 35) #35 is default of n_AA
        else:
            self.selcosts_QROM_20_cost = 3*(self.selcosts_QROM_i_cost + \
                self.selcosts_QROM_o_cost + 2) #35 is default of n_AA

        self.selcosts_QROM_cost += self.selcosts_QROM_20_cost
        return self.selcosts_QROM_cost

    def caddsub_nu_pq(self):
        self.caddsub_nu_pq_cost = 48*self.n_p
        return self.caddsub_nu_pq_cost
    
    def phasing(self):
        self.phasing_cost = 6*self.n_p*self.n_R + 12*self.n_p*self.n_R
        return self.phasing_cost

    def cost(self):
        self.cost = self.cswap() + self.selcosts_binary_QROM_decomp_cost() + self.caddsub_nu_pq() + self.phasing()
        return self.cost

class gate_cost_SlaterDet:
    def __init__(self, **kwargs):
        """
        expecting the following parameters for all methods to work:
        eta, N, N_small (optional, default set to N)
        """
        
        for param, val in kwargs.items():
            setattr(self, param, val)
        if 'N_small' not in self.__dict__.keys():
            self.N_small = self.N
        self.n_p = int(np.ceil(np.log2(self.N**(1/3) + 1)))

    def calculate_antisymmetrization_cost(self):
        # Initial state antisymmetrization
        comparison_eta = 2*(np.ceil(np.log2(self.eta**2)))
        comparison_N = 2*(np.ceil(np.log2(self.N)))
        swaps_eta = np.ceil(np.log2(self.eta**2))
        swaps_N = np.ceil(np.log2(self.N))
        Step_2 = self.eta*np.ceil(np.log2(self.eta))*(np.ceil(np.log2(self.eta))-1)/4* (comparison_eta + swaps_eta)
        Step_4 = self.eta*np.ceil(np.log2(self.eta))*(np.ceil(np.log2(self.eta))-1)/4* (comparison_N + swaps_N)
        antisymmetrization_cost = Step_2*2 + Step_4 #the *2 is due to expected success rate
        return antisymmetrization_cost
    #Pablo
 
    def cost(self): #needs to be changed to apply to adaptive sampling. unlikely to change the overall cost order
        aux = 2*(3*self.n_p-1)*self.eta # We can use m-1 Toffolis to perform a controlled not with m controls (and a few ancillas).
        swaps = (3*self.n_p)*(self.eta-1)
        Givens = 4*aux + 2*swaps # Compute and uncompute of the flag qubit, for p and q 
        HF_toffoli_cost = self.eta*(self.N_small-self.eta)*Givens + self.calculate_antisymmetrization_cost()
        HF_cost = HF_toffoli_cost
        self.cost = HF_cost
        return self.cost

class gate_cost_QPE:
    def __init__(self, **kwargs):
        """
        expecting the following parameters for all methods to work:
        lambda_val and error_qpe
        """
        for param, val in kwargs.items():
            setattr(self, param, val)

    def cost(self):
        self.cost = np.ceil(np.pi*self.lambda_val/(2*self.error_qpe))
        return self.cost

class gate_cost_PP:
    def __init__(self, **kwargs):
        self.gate_cost_QPE = gate_cost_QPE(**kwargs)
        self.gate_cost_SlaterDet = gate_cost_SlaterDet(**kwargs)
        self.gate_cost_PREP_PP = gate_cost_PREP_PP(**kwargs)
        self.gate_cost_SEL_PP = gate_cost_SEL_PP(**kwargs)
        for param, val in kwargs.items():
            setattr(self, param, val)
            if param == 'eta':
                self.n_eta = np.ceil(np.log2(self.eta))
        
        self.R0_cost = 2*self.n_eta + 9*self.n_p + \
            self.n_M_V + 35 + \
                2*(self.gate_cost_PREP_PP.n_Ltype + self.gate_cost_PREP_PP.logn_a_max) +\
                    2*1 #using a maximum of twice the Lin's trick for loc and V

    def cost(self):
        self.cost = self.gate_cost_SlaterDet.cost() + \
            self.gate_cost_QPE.cost() *(self.gate_cost_PREP_PP.cost() + \
                self.gate_cost_SEL_PP.cost() + self.R0_cost)
        return self.cost
        
    def qubit_cost(self):
        self.clean_cost = compute_clean_cost(self)
        self.dirty_cost = compute_dirty_cost(self)
        self.qubit_cost = max([self.clean_cost, self.dirty_cost])
        return self.qubit_cost

class gate_cost_HGH(gate_cost_PP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('Computing costs, this code only works for the orthogonal lattices, generalization is a WIP.')  
        self.kappa = kwargs.get("kappa")
        self.n_parallel = kwargs.get("n_parallel")
        self.n_dirty = kwargs.get("n_dirty")
        

def compute_cost_PP(error_qpe, error_op, error_R, **kwargs):
    """ 
    **kwargs must have the following parameters and their values, 
    with default values indicated in some cases below (specify PP_ params, otherwise
    won't work):
    (eta, atoms_and_rep_uc, direct_basis_vectors,
    N, error_M_V, error_Mloc, error_NL, Z_ions = None,
    PP_loc_params = None, PP_NL_params = None, amp_amp = True)
    """
    kwargs['n_p'] = np.ceil(np.log2(kwargs['N']**(1/3)+1))
    lambda_PP = lambda_X_HGH(**kwargs)
    error_X_PP = lambda_PP.error_X
    lambda_val = lambda_PP.calculate_lambda()
    print('total lambda is', lambda_val)
    kwargs["n_k"] = lambda_PP.n_k
    n_op = error_X_PP.compute_n_X(4*lambda_val, error_op)
    n_M_V, n_Mloc, n_NL = lambda_PP.n_M_V, lambda_PP.n_Mloc, lambda_PP.n_NL
    n_R = error_X_PP.compute_n_R(error_R)
    n_b = error_X_PP.compute_n_b(kwargs['error_b'])
    #below we use the LIN trick, but we don't check the condition to use it, but you can also
    #look at the print of the amp_amp probabilities for loc and V to see if LIN is applied in the right context.
    if not kwargs['qrom_loc']:
        AA_loc = min(2*lambda_PP.AA_steps_loc + 1,3) if kwargs["Lin_trick_loc"] else 2*lambda_PP.AA_steps_loc + 1
        print('2*AA_loc+1 is', AA_loc)
    else:
        AA_loc = 2
    AA_V = min(2*lambda_PP.AA_steps_V + 1, 3) if kwargs["Lin_trick_V"] else 2*lambda_PP.AA_steps_V + 1
    print('2*AA_V+1 is', AA_V)
    class_gate_cost_HGH = gate_cost_HGH(error_qpe = error_qpe, lambda_val = lambda_val, n_op = n_op,
    n_M_V = n_M_V, n_Mloc = n_Mloc, n_NL = n_NL, n_R = n_R, n_b = n_b, 
    AA_loc = AA_loc, AA_V = AA_V, natoms = error_X_PP.natoms, **kwargs)
    total_cost = class_gate_cost_HGH.cost()
    qubit_cost = class_gate_cost_HGH.qubit_cost()
    print(f'total_cost {"{:.2e}".format(Decimal(total_cost))} ; PREP {class_gate_cost_HGH.gate_cost_PREP_PP.cost}  SEL {class_gate_cost_HGH.gate_cost_SEL_PP.cost}')
    print(f'qubit cost {qubit_cost}; dirty cost {class_gate_cost_HGH.dirty_cost} clean cost {class_gate_cost_HGH.clean_cost}')
    print('parallels and dirty params are, ', kwargs["n_parallel"], kwargs["kappa"], kwargs["n_dirty"])
    print(f'results are for input N_PW = {kwargs["N"]} but computed for n_p = {lambda_PP.n_p}')
    print('finished with list of lambda values (loc, V, 11*NL, T)', lambda_PP.lambdas)

if __name__ == '__main__':
    compute_cost_PP(**Configs)