import numpy as np
import itertools

from utils import Ps
from error_X2n_X import error_X2n_X, error_X2n_X_HGH
norm = np.linalg.norm



class lambda_X:
    def __init__(self, eta, atoms_and_rep_uc, direct_basis_vectors,
     N, b_r, error_M_V, error_Mloc, error_NL, Z_ions = None, PP_loc_params = None, PP_NL_params = None, amp_amp = True, **kwargs):
        
        self.error_b = kwargs["error_b"]
        self.error_X = error_X2n_X(eta, atoms_and_rep_uc, direct_basis_vectors,
     N,  Z_ions, PP_loc_params, PP_NL_params, **kwargs)
        
        for param, val in self.error_X.__dict__.items():
            if param != 'self':
                setattr(self, param, val)
        self.kwargs = kwargs
        self.error_k =  self.kwargs["error_k"]
        self.qrom_loc = self.kwargs["qrom_loc"]
        self.pnuth = self.kwargs["pnuth"]
        self.material_ortho_lattice = kwargs["material_ortho_lattice"]
        self.b_r = b_r
        self.amp_amp = amp_amp
        self.error_M_V = error_M_V
        self.error_Mloc = error_Mloc
        self.error_NL = error_NL
        

   
    def compute_AA_steps(self, pnu, th = None):
        th = self.pnuth if th is None else th
        amplitude_amplified = 0
        index = 0
        for i in range(29,-1,-1):
            amplitude = (np.sin((2*i+1)*np.arcsin(np.sqrt(pnu))))**2
            if amplitude>th:
                index = i
                amplitude_amplified = amplitude
        pnu_amps = [(np.sin((2*i+1)*np.arcsin(np.sqrt(pnu))))**2 for i in range(6)]
        # if pnu_amps[0] < 0.25 and pnu_amps[1] > th:
        #     amplitude_amplified, index = pnu_amps[1]**2, 1
        # elif pnu_amps[0] > np.sin(np.pi/10)**2 and pnu_amps[1]  th:
        print(pnu, pnu_amps)
        return amplitude_amplified, index

    #Pablo    
    def calculate_lambda_T(self):
        self.abs_sum = 0 
        for b_omega in self.recip_bv:
            for b_omega_prime in self.recip_bv:
                self.abs_sum += np.abs(np.sum(b_omega*b_omega_prime))
        # self.lambda_T = self.eta/2 * self.abs_sum * \
        #         2**(2*self.n_p-2)/(1-(4*np.pi/2**(self.error_X.compute_n_b(self.error_b)))**2)
        self.lambda_T = self.eta/2 * self.abs_sum * \
                2**(2*self.n_p-2)
        if self.material_ortho_lattice:
            self.lambda_T /= 2    
        return self.lambda_T
    
    def compute_B_mus(self): 
        B_mus = {}
        for j in range(2, self.n_p+3):
            B_mus[j] = []
        for nu in itertools.product(range(-2**(self.n_p), 2**(self.n_p)+1), repeat = 3):
            nu = np.array(nu)
            if list(nu) != [0,0,0]:
                mu = int(np.floor(np.log2(np.max(abs(nu)))))+2
                B_mus[mu].append(nu)
        self.B_mus = B_mus


    def compute_lambda_V_nu_one(self):
        self.n_M_V = self.error_X.compute_n_M_V(self.error_M_V)    
        if 'B_mus' not in self.__dict__.keys():
            self.compute_B_mus()
        self.M_V = 2**self.n_M_V
        lambda_nu_one = 0
        p_nu_one = 0
        for mu in range(2, (self.n_p+2)):
            for nu in self.B_mus[mu]:
                Gnu_norm = norm(self.error_X.G_p(nu))
                p_nu_one += np.ceil( self.M_V*(self.bmin* 2**(mu-2)/Gnu_norm)**2)/\
                        (self.M_V * (2**(mu-2))**2)
                lambda_nu_one += np.ceil( self.M_V*(self.bmin*2**(mu-2)/Gnu_norm)**2)/\
                        (self.M_V * (self.bmin*2**(mu-2))**2)
        self.lambda_nu_one = lambda_nu_one  
        self.p_nu_one = p_nu_one
    
    def compute_lambda_V_one(self):
        self.compute_lambda_V_nu_one()
        self.lambda_V_one = 2*np.pi*self.eta*(self.eta-1)*self.lambda_nu_one/self.Omega
    
    def compute_lambda_nu_loc_one(self):
        scalar_factor_n_Mloc = 0
        for idx, atom in enumerate(self.list_atoms):
            I = self.PP_loc_params[atom]
            scalar_factor_n_Mloc += Ps(self.atoms_rep[idx], self.b_r)*Ps(self.natoms_type, self.b_r)
        
        self.n_Mloc = self.error_X.compute_n_Mloc(self.error_Mloc,scalar_factor_n_Mloc)
        
        self.Mloc = 2**self.n_Mloc
        if 'k_locmin_val' not in self.error_X.__dict__.keys():
            self.error_X.k_locmin()
        self.k_locmin_val = self.error_X.k_locmin_val
        if 'B_mus' not in self.__dict__.keys():
            self.compute_B_mus()
        lambda_nu_loc_one = 0
        p_nu_loc_one = 0
        for mu in range(2, (self.n_p+2)):
            for nu in self.B_mus[mu]:
                Gnu_norm = norm(self.error_X.G_p(nu))
                for idx, atom in enumerate(self.list_atoms):
                    I = self.PP_loc_params[atom]
                    k_loc_val = self.error_X.k_loc(Gnu_norm,I)
                    lambda_nu_loc_one += np.ceil( self.Mloc*self.atoms_rep[idx]*\
                        np.abs(k_loc_val)/(Ps(self.natoms_type, self.b_r) * Ps(self.atoms_rep[idx], self.b_r))* \
                        (self.k_locmin_val*self.bmin*2**(mu-2) / Gnu_norm)**2)/\
                            (self.Mloc * ( self.k_locmin_val*self.bmin*2**(mu-2) )**2)
                    p_nu_loc_one += np.ceil( self.Mloc*self.atoms_rep[idx]*\
                        np.abs(k_loc_val)/(Ps(self.natoms_type, self.b_r) * Ps(self.atoms_rep[idx], self.b_r))*\
                        ( self.k_locmin_val*self.bmin* 2**(mu-2) / Gnu_norm)**2)/\
                            (self.Mloc * ( 2**(mu-2) )**2)
        self.lambda_nu_loc_one = lambda_nu_loc_one
        self.p_nu_loc_one = p_nu_loc_one*Ps(self.natoms_type,self.b_r)/self.natoms_type

    def compute_lambda_loc_one(self):
        self.compute_lambda_nu_loc_one()
        self.lambda_loc_one = 4*np.pi*self.eta*self.lambda_nu_loc_one/self.Omega


    
    def compute_p_nu_V(self):
        if 'lambda_nu_one' not in self.__dict__.keys():
            self.compute_lambda_nu_one()
        self.p_nu_V = self.p_nu_one/2**(self.n_p+6)      

    def compute_p_nu_amp_V(self):
        if 'p_nu_V' not in self.__dict__.keys():
            self.compute_p_nu_V()
        self.p_nu_amp_V, self.AA_steps_V = self.compute_AA_steps(self.p_nu_V)


    def compute_p_nu_loc(self):
        if 'lambda_nu_loc_one' not in self.__dict__.keys():
            self.compute_lambda_nu_loc_one()
        self.p_nu_loc = self.p_nu_loc_one/2**(self.n_p+6)
    
    def compute_p_nu_amp_loc(self):
        if 'p_nu_loc' not in self.__dict__.keys():
            self.compute_p_nu_loc()
        self.p_nu_amp_loc, self.AA_steps_loc = self.compute_AA_steps(self.p_nu_loc)

    def compute_lambda_NL(self): #PP specific implementation
        self.lambda_NL = []
    
    def compute_n_NL(self):#PP specific implementation
        pass

    def compute_lambda_loc(self):
        self.compute_lambda_loc_one()
        print('loc AA probabilities')
        if not self.qrom_loc:
            self.compute_p_nu_amp_loc()
            print('the amplified prob ', self.p_nu_amp_loc)
            self.lambda_loc = self.lambda_loc_one/self.p_nu_amp_loc
        
    def compute_lambda_V(self):
        self.compute_lambda_V_one()
        print('V AA probabilities')
        self.compute_p_nu_amp_V()
        self.lambda_V = self.lambda_V_one/self.p_nu_amp_V

    def calculate_lambdas(self):
        self.compute_lambda_loc()
        self.compute_lambda_V()        
        self.compute_lambda_NL()
        self.calculate_lambda_T()
        self.lambdas = [self.lambda_loc,  self.lambda_V] +\
            self.lambda_NL +[self.lambda_T]

    def calculate_lambda(self):
        self.calculate_lambdas()
        
        self.lambda_val = sum(self.lambdas)/Ps(self.eta, self.b_r)**2
        return self.lambda_val
    #PP Amplitude Amplifications calculations


class lambda_X_HGH(lambda_X):
    def __init__(self, material = 'Li_2FeSiO_4', **kwargs):
        super().__init__(**kwargs)
        self.material = material
        self.error_X = error_X2n_X_HGH(self.material, **kwargs)
        self.error_X.b_r = kwargs["b_r"]
        if self.qrom_loc: self.compute_lambda_loc_one = self.compute_lambda_loc_one_qrom
        
    def compute_lambda_loc_one_qrom(self):
        self.integral_loc = 0
        self.integral_loc_error = 0
        #below we have not the best bound for error! wish I had sth better!
        maxnaPa = np.max([self.atoms_rep[idx]/Ps(self.atoms_rep[idx], self.b_r) for idx,_ in enumerate(self.list_atoms)])
        for idx, atom in enumerate(self.list_atoms):
            I = self.PP_loc_params[atom]
            X_I = np.sum([self.atoms_rep[idx]*abs(self.error_X.k_loc(Gnu_norm,I))/(Ps(self.atoms_rep[idx],self.b_r) * Gnu_norm**2 )\
                if abs(Gnu_norm)>1e-7 else 0 \
                  for Gnu_norm in self.error_X.G_pnorms])
            self.integral_loc += X_I
            self.integral_loc_error += (X_I*Ps(self.atoms_rep[idx],self.b_r)/self.atoms_rep[idx])
        scalar = 4*np.pi*self.eta/self.Omega 
        self.n_Mloc = self.error_X.error2n(scalar * self.integral_loc_error * 2*np.pi*maxnaPa*(3*self.n_p+self.n_Ltype)/self.error_Mloc)
        
        # self.lambda_loc = scalar * self.integral_loc / (1-(4*np.pi/(2**self.n_Mloc))**2) #this one is right and triple checked
        #* (1 + 2**(-self.n_Mloc)*np.pi*(3*self.n_p+self.n_Ltype))
        self.lambda_loc = scalar * self.integral_loc #this one is right and triple checked


    def compute_lambda_NL(self): #HGH specific implementation
        NL0 = np.array([0.]*len(self.list_atoms))
        NL1omega = np.array([[0.]*3]*len(self.list_atoms))
        NL20 = np.array([0.]*len(self.list_atoms))
        NL2omegaomega = np.array([[0.]*6]*len(self.list_atoms))
        self.error_X.G_psomega = [np.array([Gp[omega] for Gp in self.error_X.G_ps]) for omega in range(3)]
        self.error_X.G_psomegasquared = [x**2 for x in self.error_X.G_psomega]
        self.error_X.G_psomegaomega = [self.error_X.G_psomega[i]* \
            self.error_X.G_psomega[j]  for i,j in [(0,0),(1,1),(2,2),(0,1),(1,2),(2,1)]]
        self.error_X.G_psomegaomegasquared = [x**2 for x in self.error_X.G_psomegaomega]
        self.error_X.G_pnormsquad = self.error_X.G_pnorms**4

        
        multipliers = [{},{},{},{}]

        for idx, atom in enumerate(self.list_atoms):
            I = self.PP_NL_params[atom]
            rs, Bi_inv = I['rs'], I['Bi_inv']
            
            multipliers[0][atom] = (self.error_X.G_pnormsexps**(rs[0]**2)).sum()
            # NL0[idx] = self.atoms_rep[idx] * rs[0]**3 * abs(Bi_inv[0]) * \
            #     multipliers[0][atom]/Ps(self.atoms_rep[idx],self.b_r)
            NL0[idx] = self.atoms_rep[idx] * rs[0]**3 * abs(Bi_inv[0]) * \
                multipliers[0][atom]
            multipliers[1][atom] = []
            for i in range(3):
                multipliers[1][atom].append((self.error_X.G_psomegasquared[i] *\
                    self.error_X.G_pnormsexps**(rs[1]**2)).sum())
                # NL1omega[idx][i] = self.atoms_rep[idx] * rs[1]**5 * abs(Bi_inv[1]) * \
                #     multipliers[1][atom][-1]/Ps(self.atoms_rep[idx],self.b_r)
                NL1omega[idx][i] = self.atoms_rep[idx] * rs[1]**5 * abs(Bi_inv[1]) * \
                    multipliers[1][atom][-1]
            multipliers[2][atom] = (self.error_X.G_pnormsquad * \
                self.error_X.G_pnormsexps**(rs[2]**2)).sum()
            # NL20[idx] =  self.atoms_rep[idx] * rs[2]**7 * abs(Bi_inv[2]) * \
            #     multipliers[2][atom]/Ps(self.atoms_rep[idx],self.b_r)
            NL20[idx] =  self.atoms_rep[idx] * rs[2]**7 * abs(Bi_inv[2]) * \
                multipliers[2][atom]
            
            multipliers[3][atom] = []
            for i in range(6):
                multipliers[3][atom].append((self.error_X.G_psomegaomegasquared[i] * \
                    self.error_X.G_pnormsexps**(rs[2]**2)).sum())
                # NL2omegaomega[idx][i] = (int(i>2)+1) * self.atoms_rep[idx] * rs[2]**7 * \
                #     abs(Bi_inv[2]) * multipliers[3][atom][-1]/Ps(self.atoms_rep[idx],self.b_r)
                NL2omegaomega[idx][i] = (int(i>2)+1) * self.atoms_rep[idx] * rs[2]**7 * \
                    abs(Bi_inv[2]) * multipliers[3][atom][-1]

        self.multipliers = multipliers
        NL0 = NL0 * 8*np.pi*self.eta/self.Omega
        NL1omega = NL1omega * 32*np.pi*self.eta/(3*self.Omega)
        NL20 = NL20 * 64*np.pi*self.eta/(45*self.Omega)
        NL2omegaomega = NL2omegaomega * 64*np.pi*self.eta/(15*self.Omega)

        NL0sum = NL0.sum()
        NL1omegasum = list(NL1omega.sum(axis=0))
        NL20sum = NL20.sum()
        NL2omegaomegasum = list(NL2omegaomega.sum(axis=0))
        self.lambda_NL_prime = [NL0sum] + NL1omegasum + [NL20sum] + NL2omegaomegasum
        self.n_k = self.error_X.compute_n_k(self.error_k, np.array(self.lambda_NL_prime))    
        # self.lambda_NL = [x/(1-((4+self.n_Ltype)*np.pi/2**(self.n_k))**2)\
        #     for x in self.lambda_NL_prime]
        self.lambda_NL = [x for x in self.lambda_NL_prime]
        self.compute_n_NL() #this is to be used later on

    def compute_n_NL(self):
        self.n_NL = self.error_X.compute_n_NL(self.error_NL, self.multipliers)
