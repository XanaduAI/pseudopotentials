#############################################################################
#An adaptation of [T-Fermion](https://github.com/PabloAMC/TFermion) to materials with nonorthogonal lattice
#Notations and equations follow the accompanying paper and [Su. et. al.](https://arxiv.org/pdf/2105.12767.pdf)
#############################################################################

import numpy as np
from utils import *
import itertools

def compute_lambda_nu(n_p, M, bmin, B_mus, recip_bv, lambda_nu_orthonormal):  
  lambda_nu = 0
  if n_p <=6:
    for mu in range(2, (n_p+2)):
        for nu in B_mus[mu]:
          Gnu_norm = np.linalg.norm(np.sum(nu * recip_bv,axis = 0))
          lambda_nu += np.ceil( M*( bmin*2**(mu-2)/Gnu_norm)**2)/\
                  (M * (bmin*2**(mu-2))**2)
  else: #approximation for n_p >=7, exact calculation like above takes too long
    lambda_nu = lambda_nu_orthonormal/bmin**2
  return lambda_nu


def calculate_lambdas(N, eta, lambda_zeta, Omega, n_p, M, bmin, B_mus, recip_bv, n_B, material_ortho_lattice):

    
    lambda_nu_orthonormal = 4*np.pi*(np.sqrt(3)*N**(1/3)/2 - 1) + 3 - 3/N**(1/3) + 3*I(N**(1/3))
    abs_sum = 0
    for b_omega in recip_bv:
      for b_omega_prime in recip_bv:
          abs_sum += np.abs(np.sum(b_omega*b_omega_prime))

    if material_ortho_lattice:
      lambda_prime_T = abs_sum*eta* 2**(2*n_p-2) /(4)
    else:
      lambda_prime_T = abs_sum*eta* 2**(2*n_p-2) /(2)
    eps = 4/(M*bmin**2) *(7*2**(n_p+1) + 9*n_p -11 -3*2**(-n_p))
    lambda_nu = compute_lambda_nu(n_p,M, bmin, B_mus, recip_bv, lambda_nu_orthonormal)
    if n_p<= 6: 
      lambda_nu_adjusted = lambda_nu
    else:
      lambda_nu_adjusted = lambda_nu*(1+eps/lambda_nu)
    # print('lambda_nu', lambda_nu)
    lambda_V_prime = 2*np.pi*eta*(eta-1)*lambda_nu_adjusted/Omega
    lambda_U_prime = 4*np.pi*eta*lambda_zeta*lambda_nu_adjusted/Omega

    return lambda_U_prime, lambda_V_prime, lambda_prime_T, lambda_nu,lambda_nu_adjusted,eps

def calculate_number_bits_parameters(optimized_parameters, N, n_p, eta, lambda_zeta, Omega, recip_bv, B_mus, bmin, pth, epsilon_B, material_ortho_lattice):

    _, epsilon_M, epsilon_R, _, epsilon_T, br = optimized_parameters
    Peq = Ps(eta+2*lambda_zeta,br)*(Ps(eta, br))**2 #for general lattice the T superposition is done via QROM
    # n_eta
    n_eta = np.ceil(np.log2(eta))
    # n_eta_zeta
    n_eta_zeta = np.ceil(np.log2(eta+2*lambda_zeta))
    # n_M
    n_M = np.ceil(np.log2( (8*np.pi*eta)*(eta-1+2*lambda_zeta)*(7*2**(n_p+1)-9*n_p-11-3*2**(-n_p))/(epsilon_M* Omega * bmin**2)))
    M  = 2**n_M
    # n_R
    n_R = np.ceil(np.log2( eta*lambda_zeta/(epsilon_R*Omega**(1/3))*sum_1_over_nu(N)))
    # n_B
    def compute_n_b(error_b):
      abs_sum = sum([np.abs(np.sum(b1*b2)) for b1,b2 in itertools.product(recip_bv, repeat=2)])
      scalar = 2*np.pi*eta*2**(2*n_p-2)*abs_sum/error_b
      return np.ceil(np.log2(scalar))
    n_B = compute_n_b(epsilon_B)

    lambda_U_prime, lambda_V_prime, lambda_prime_T,  lambda_nu, lambda_nu_adjusted, eps = calculate_lambdas(N, eta, lambda_zeta, Omega, n_p, M, bmin, B_mus, recip_bv, n_B, material_ortho_lattice)
    p_nu_U, p_nu_amp_U, a_U = compute_p_nu(n_p, M, recip_bv, B_mus, bmin, pth, lambda_nu_adjusted)
    
    # uncomment below to see values of lambdas
    # print('lambda_U_prime, lambda_prime_T, lambda_V_prime: ', lambda_U_prime , lambda_prime_T, lambda_V_prime)
    #Effective lambda depends on how lambda_prime_T compares with the rest (see Theorem 4 in https://arxiv.org/pdf/2105.12767.pdf)
    if p_nu_U * lambda_prime_T >= (1-p_nu_U)*(lambda_U_prime+lambda_V_prime):
      # print('SEL_T trick without AA')
      lambda_value = lambda_prime_T + lambda_U_prime + lambda_V_prime
      a_U = 0
    elif p_nu_amp_U * lambda_prime_T >= (1-p_nu_amp_U)*(lambda_U_prime+lambda_V_prime):
      # print('SEL_T trick with AA')
      lambda_value = lambda_prime_T+lambda_U_prime+lambda_V_prime
    else:
      # print('SEL_T worst case trick')
      lambda_value = (lambda_U_prime+lambda_V_prime/(1-1/eta))/p_nu_amp_U
    lambda_value /= Peq
    n_T = np.ceil(np.log2( np.pi*lambda_value/epsilon_T ))
    return n_p, n_eta, n_eta_zeta, n_M, n_R, n_T, n_B, lambda_value, a_U