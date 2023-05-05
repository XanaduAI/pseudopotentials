import numpy as np
import itertools   
from scipy import integrate
from utils import Ps
norm = np.linalg.norm

class error_X2n_X:
    def __init__(self, eta, atoms_and_rep_uc, direct_basis_vectors,
     N,  PP_loc_params = None, PP_NL_params = None , **kwargs):
        self.N = N
        self.n_p = int(np.ceil(np.log2(N**(1/3) + 1)))
        self.eta = eta
        self.list_atoms, self.atoms_rep_uc = list(atoms_and_rep_uc.keys()), np.array(list(atoms_and_rep_uc.values()))
        
        self.Z_ions = np.array([PP_loc_params[k]['Z_ion'] for k in self.list_atoms])        
        self.PP_loc_params = PP_loc_params
        self.PP_NL_params = PP_NL_params

        self.direct_bv = np.array(direct_basis_vectors)
        
        self.angs2bohr = 1.8897259886
        self.maxai = max([norm(ai) for ai in self.direct_bv]) * self.angs2bohr
        def compute_Omega(vecs):
            return np.abs(np.sum((np.cross(vecs[0],vecs[1])*vecs[2]))) * self.angs2bohr**3  # | (a cross b) . c |  = volume of parallelepiped

        self.Omega = compute_Omega(self.direct_bv)
        # print(f'Omega in bohr {self.Omega} in angs {self.Omega/ self.angs2bohr**3}')
        self.atoms_rep = self.atoms_rep_uc
        self.lambda_zeta = (self.Z_ions*self.atoms_rep).sum()
        self.natoms = self.atoms_rep.sum()
        self.angs2bohr = 1.8897259886
        self.recip_bv = 2*np.pi/self.Omega * \
            np.array([np.cross(self.direct_bv[i],self.direct_bv[j]) for i,j in [(1,2),(2,0),(0,1)]]) * self.angs2bohr**2
        recip_bv_u, recip_bv_s, recip_bv_v = np.linalg.svd(self.recip_bv)
        self.lambda_min_B , self.lambda_max_B = np.min(recip_bv_s), np.max(recip_bv_s) 
        self.bmin = self.lambda_min_B
        # self.bmin2 = np.min([norm(bi) for bi in self.recip_bv])
        # assert abs(self.bmin2 - self.bmin) < 1e-6 #comment this out and you will see the assert goes thru --> The two nonorthogonal case studies have bmin = min ||b_i||    
        self.mcG = self.G_lattice(self.n_p)
        self.G_ps = np.array([self.G_p(p) for p in self.mcG])
        self.G_pnorms = np.array([norm(G_p) for G_p in self.G_ps])
        self.G_pnormsexps = np.array([np.exp(-nGp**2) for nGp in self.G_pnorms])
        self.n_L = np.ceil(np.log2(self.natoms))
        self.natoms_type = len(self.list_atoms)
        self.n_Ltype = np.ceil(np.log2(self.natoms_type))
        self.material_ortho_lattice = kwargs['material_ortho_lattice']
        
    def error2n(self, inv_error):
        return np.ceil(np.log2(inv_error))
    
    def compute_n_X(self, lambda_total, error_X):
        scalar = np.pi*lambda_total
        integral = 1/error_X
        return self.error2n(scalar*integral)

    def compute_n_R(self, error_R):
        scalar = 2*self.eta*np.pi*self.maxai/(error_R*self.Omega)
        integral = self.sum_kloc_over_Gnu(self.n_p) + self.sum_kNL_over_Gnu(self.n_p)
        return self.error2n(scalar*integral)
        
    def compute_n_M_V(self, error_M_V): #n_M_V
        scalar = 8*np.pi*self.eta*(self.eta-1)/(self.Omega* self.bmin**2)
        integral = (7*2**(self.n_p+1)-9*self.n_p-11-3*2**(-self.n_p))/error_M_V
        return self.error2n(scalar*integral)
    
    
    
    def compute_n_b(self, error_b):
        self.abs_sum = 0 
        for b_omega in self.recip_bv:
            for b_omega_prime in self.recip_bv:
                self.abs_sum += np.abs(np.sum(b_omega*b_omega_prime))
        scalar = 2*np.pi*self.eta*2**(2*self.n_p-2)*self.abs_sum/error_b
        if not self.material_ortho_lattice:
            scalar *= 2
        if 'sum_1_over_Gnu_squared_result' not in self.__dict__.keys():
            self.sum_1_over_Gnu_squared(self.n_p)
        integral = 1
        return self.error2n(scalar*integral) 

    def sum_1_over_Gnu_squared(self, n_p):
        self.sum_1_over_Gnu_squared_result = np.sum([1/normGp**2 if abs(normGp)>1e-7 else 0 for normGp in self.G_pnorms])
        return self.sum_1_over_Gnu_squared_result

    def sum_1_over_Gnu(self, n_p):
        return np.sum([1/normGp if abs(normGp)>1e-7 else 0 for normGp in self.G_pnorms])

    def f(self, x, y):
        return 1/(x**2 + y**2)
    def Intnquad(self, N0):
        return integrate.nquad(self.f, [[1, N0],[1, N0]])[0]
    def sum_1_over_nu_squared_fast(self, N):
        return 4*np.pi*(np.sqrt(3)*N**(1/3)/2 - 1) + 3 - 3/N**(1/3) + 3*self.Intnquad(N**(1/3))

    def G_lattice(self, n_p):
        return list(itertools.product(range(-2**(n_p-1)+1, 2**(n_p-1)), repeat = 3))

    def G_p(self, p):
        return np.sum(np.array(p)*self.recip_bv, axis=0)

    def sum_kloc_over_Gnu(self, n_p):
        mcG = self.mcG if n_p == self.n_p else self.G_lattice(n_p)
        integral = 0        
        integral2 = 0
        for nu in mcG:
            if np.all(nu)!=0:
                for idx, atom in enumerate(self.list_atoms):
                    I = self.PP_loc_params[atom]
                    nxt= self.atoms_rep[idx]*np.abs(self.k_loc(norm(self.G_p(nu)),I))/norm(self.G_p(nu))                    
                    integral += nxt
                    integral2 += nxt/norm(self.G_p(nu))
        self.sum_kloc_over_Gnu_squared_result = integral2
        return integral
        
    def sum_kNL_over_Gnu(self, n_p, fast = True):
        mcG = self.mcG if n_p == self.n_p else self.G_lattice(n_p)
        integral = 0
        if fast:
            for idx, atom in enumerate(self.list_atoms):
                I = self.PP_NL_params[atom]        
                integral += self.atoms_rep[idx]*np.abs(self.k_NL_fast(I))
        else:
            self.Gsq = list(itertools.product(mcG,repeat=2))
            for p,q in self.Gsq:
                if p != q:
                    for idx, atom in enumerate(self.list_atoms):
                        I = self.PP_NL_params[atom]
                        integral += self.atoms_rep[idx]*np.abs(self.k_NL(self.G_p(p), self.G_p(q),I))/norm(self.G_p(p)-self.G_p(q))
        return integral
    
    #PP specific
    def k_loc(self, G_norm, I, *args, **kwargs): #called gamma in the paper
        return 0
    def k_NL(self, G_p, G_q, I, *args, **kwargs):
        return 0
    def k_NL_fast(self, Gsq, *args, **kwargs):
        return 0
    def compute_n_NL(self, error_NL):
        pass



class error_X2n_X_HGH(error_X2n_X):
    def __init__(self, material = 'Li_2FeSiO_4', **kwargs):
        super().__init__(**kwargs)
        self.material = material

    def compute_n_NL(self, error_NL, multipliers):        
        atom_n_NL_error = 0
        for idx, atom in enumerate(self.list_atoms):
            I = self.PP_NL_params[atom]
            rs, Bi_inv = I['rs'], I['Bi_inv']
            atom_n_NL_error += rs[0]**3 * abs(Bi_inv[0]) * multipliers[0][atom]
            atom_n_NL_error += 32/3 * rs[1]**5 * abs(Bi_inv[1]) * sum(multipliers[1][atom])
            atom_n_NL_error += 64/45 * rs[2]**7 * abs(Bi_inv[2]) * multipliers[2][atom]
            atom_n_NL_error += 64/15 * rs[2]**7 * abs(Bi_inv[2]) * sum(multipliers[3][atom])
            atom_n_NL_error *= self.atoms_rep[idx]
        scalar = 18 * (self.n_p + 4 + self.n_Ltype) * np.pi**2 * self.eta *  atom_n_NL_error / (self.Omega * error_NL) 
        if not self.material_ortho_lattice:
            scalar = scalar * (2*self.n_p + 4 + self.n_Ltype)/(self.n_p + 4 + self.n_Ltype)
        return self.error2n(scalar)

    def compute_n_k(self, error_k, lambda_NL_prime):
        return self.error2n(2*(self.n_Ltype+4)*np.pi*sum(lambda_NL_prime)/error_k)

    def k_loc(self, G_norm, I): #I contains Z_ion, r_loc, Cs, this is function gamma in the paper
        Z_ion, r_loc, Cs = I['Z_ion'], I['r_loc'], I['Cs']
        D1 = (Cs[0]+3*Cs[1])*np.sqrt(np.pi)*r_loc**3/2
        D2 = Cs[1]*np.sqrt(np.pi)*r_loc**5/2
        return np.exp(-(G_norm*r_loc)**2/2)*(-Z_ion + (D1-D2*G_norm**2)*G_norm**2)
            
    
    def k_NL(self, G_p, G_q, I): #I contains r0, r1, r2, B0_inv, B1_inv, B2_inv
        rs, Bi_inv = I['rs'], I['Bi_inv']
        nGp, nGq = norm(G_p), norm(G_q)
        x = -(nGp**2 + nGq**2)/2
        ex = np.exp(x)
        gpdotgq = np.dot(G_p,G_q)

        fpq = 4*rs[0]**3 * Bi_inv[0] * ex**(rs[0]**2) + \
            16/3 * rs[1]**5 * Bi_inv[1] * gpdotgq * ex**(rs[1]**2)  + \
            32/15 * rs[2]**7 * Bi_inv[2]*(gpdotgq**2 + 1/3 * nGp**2 * nGq**2)* ex**(rs[2]*2)
        return norm(G_p-G_q)**2 * fpq
            
    def k_NL_fast(self, I): #based on |G_p-G_q| * fpq <= (|G_p| + |G_q|)fpq = ...
        rs, Bi_inv = I['rs'], I['Bi_inv']

        a0 = abs(4*rs[0]**3 * Bi_inv[0])
        exps0 = self.G_pnormsexps**(rs[0]**2/2)
        nGpexp0 = np.sum(self.G_pnorms * exps0)
        exp0 = np.sum(exps0)
        nondiagest0 = nGpexp0 * exp0 - np.sum(self.G_pnorms * exps0**2)
        Int0 = 2* a0 * nondiagest0

        a1 = abs(16/3 * rs[1]**5 * Bi_inv[1])
        exps1 = self.G_pnormsexps**(rs[1]**2/2)
        nGpexp1 = np.sum(self.G_pnorms * exps1)
        nGp2exp1 = np.sum(self.G_pnorms**2 * exps1)
        nondiagest1 = nGpexp1*nGp2exp1 - np.sum(self.G_pnorms**3 * exps1**2)
        Int1 = 2 * a1 * nondiagest1
        
        a2 = abs(128/45 * rs[2]**7 * Bi_inv[2])
        exps2 = self.G_pnormsexps**(rs[2]**2/2)
        nGp3exp2 = np.sum(self.G_pnorms**3 * exps2)
        nGp2exp2 = np.sum(self.G_pnorms**2 * exps2)
        nondiagest2 = nGp3exp2 * nGp2exp2 - np.sum(self.G_pnorms**5 * exps2**2)
        Int2 = 2* a2 * nondiagest2
        return Int0 + Int1 + Int2
            