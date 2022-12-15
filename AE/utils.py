
import numpy as np
from scipy import integrate



def compute_p_nu(n_p, M, recip_bv, B_mus, bmin, pth, lambda_nu_adjusted):

    n_M = np.log2(M)

    print('n_M',n_M)
    print('n_p',n_p)
    print('recip_bv', recip_bv)
    p_nu = 0
    # if n_p <= 6:
    #     for mu in range(2, (n_p+2)):
    #         for nu in B_mus[mu]:
    #             Gnu_norm = np.linalg.norm(np.sum(nu * recip_bv,axis = 0))
    #             p_nu += np.ceil( M*( bmin* 2**(mu-2) / Gnu_norm )**2)/(M*2**(2*mu)*2**(n_p+2))
    # else:
    p_nu = lambda_nu_adjusted*bmin**2/2**(n_p+6)

    print('p_nu', p_nu)
    p_nu_amp, a_U = compute_AA_steps(p_nu, pth)
    return p_nu, p_nu_amp, a_U

def compute_AA_steps(pnu, th):
    amplitude_amplified = 0
    index = 0
    for i in range(29,-1,-1):
        amplitude = (np.sin((2*i+1)*np.arcsin(np.sqrt(pnu))))**2
        if amplitude>th:
            index = i
            amplitude_amplified = amplitude
    print(pnu, [(np.sin((2*i+1)*np.arcsin(np.sqrt(pnu))))**2 for i in range(6)])
    return amplitude_amplified, index


"""Auxiliary functions"""

def sum_cost(n):
    return 4*n
def pauli_rotation_synthesis(epsilon_SS):
    result = 10 + 4*np.ceil(np.log2(1/epsilon_SS))
    return result
def c_pauli_rotation_synthesis(epsilon_SS):
    return 2*pauli_rotation_synthesis(epsilon_SS)
def compare_cost(n):
    return 8*n
def sum_1_over_nu(N):
    return 2*np.pi*N**(2/3) # based on integrate(r*sin(t), (r, 0, N), (p, 0, 2*pi), (t, 0, pi)) and eq 13 in https://www.nature.com/articles/s41534-019-0199-y
def f(x, y):
    return 1/(x**2 + y**2)
def I(N0):
    return integrate.nquad(f, [[1, N0],[1, N0]])[0]

def Er(x):
    logx = np.log2(x)
    fres = 2**(np.floor(logx/2)) + np.ceil(2**(-np.floor(logx/2))*x)
    cres = 2**(np.ceil(logx/2)) + np.ceil(2**(-np.ceil(logx/2))*x)
    return min(fres, cres)

def Ps(n, br): #eq 59 from https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040332

        theta = 2*np.pi/(2**br)*np.round((2**br)/(2*np.pi)*np.arcsin(np.sqrt(2**(np.ceil(np.log2(n)))/(4*n)))) #eq 60
        braket = (1+(2-(4*n)/(2**np.ceil(np.log2(n))))*(np.sin(theta))**2)**2 + (np.sin(2*theta))**2
        return n/(2**np.ceil(np.log2(n)))*braket