<div align="center">     
 [![Paper](http://img.shields.io/badge/arxiv-quant.ph:2110.05899-B31B1B.svg)](https://arxiv.org/abs/X)
 <!-- ![Paper](http://img.shields.io/badge/2022-Quantum-purple.svg)](https://quantum-journal.org/papers/q-2022-07-20-768/) -->
</div>

# Resource Estimation of Qubitization-based QPE

## How to install  
Clone and install dependencies   
```bash
# clone project   
git clone https://github.com/XanaduAI/pseudopotentials.git

# install auxiliary libraries
pip install fire numpy scipy
 ```   

## What can this code do
The code estimating the resources for the pseudopotential-based algorithm (`PP/`) works for the HGH pseudopotential; However, if the user has a different pseudopotential and algorithm to qubitize the potential terms, we have made it easy to change the code for those terms.

A limitation of the PP code is that it runs for a number `N` of plane waves (PWs) that is at most `2**18 ~ 2.62e5`. Going beyond that requires approximations that make the resource estimation less accurate, and is also arguably unnecessary, as pseudopotential algorithms for materials of interest (not too large and not too small) generally reach convergence using an `N` with the same order of magnitude. 

However the all-electron calculations (`AE/`) must be done for a larger `N`, as convergence of the algorithm can only be achieved for a much higher number of PWs, which this code supports.

We can do resource estimation for the following materials:
```
dilithium_iron_silicate = Li2FeSiO4
Li_rich_manganese_oxyfluoride = Li0.5MnO2F
LLNMO = Li0.75[Li0.17Mn0.58Ni0.25]O2
Li_rich_manganese_oxide= Li0.5MnO3
```
and we index these materials with `mat_idx = 0, 1, 2, 3` respectively. To define a new material, the user has to define it in the `PP/Configs.py` and `AE/cost.py` files. We refer to those files and the accompanying paper to see exactly what properties of the material are asked for and how they can be obtained. Finally, in line with the accompanying paper, we note that the AE code supports the resource estimation for materials with any lattice structure, while the PP code supports it for those with a partially orthogonal lattice (when a lattice vector is orthogonal to the other two). 

## How to run
Inside the repository, run the following to estimate the resources of the pseudopotential-based algorithm:
```
~/pseudopotentials> python PP/cost.py  -N [# of plane waves] -mat_idx [material's index] -error [target error] -n_tof [max # of simultaneous Toffolis] -p_th [amplitude amplification threshold]
```
Similarly, for the AE algorithm:
```
~/pseudopotentials> python AE/cost.py -N [# of plane waves] -mat_idx [material's index] -error [target error] -n_tof [max # of simultaneous Toffolis]  -p_th [amplitude amplification threshold] 
```
To optimize the depth of the algorithm, choose any value `n_tof > 1` (default is `500`), and for optimizing cost, enter `-n_tof  1`. Refer to the accompanying paper for the default values of other parameters.

The code starts by outputting all the parameters taken as input, and finishes by outputting the results:
```bash
~/pseudopotentials> python PP/cost.py -N 1e4  -mat_idx 0 -error 1.5e-3 -n_tof 500 -p_th 0.75
`[... All input parameters ...]`
lambda  125190.6543226223
Total Toffoli 1.91e+13 PREP 91261.0 SEL 53850.0
Qubit cost 2847.0 clean 2620.0
```
`lambda` denotes the LCU-induced one-norm of the Hamiltonian, `Total Toffoli` is the optimized Toffoli cost/depth depending on the value of `n_tof`, and `PREP` and `SEL` are the optimized Toffoli cost/depth of the qubitization subroutines. `Qubit cost` is the entire qubit cost of the algorithm, and `clean` is the clean qubit cost. Any difference is due to the QROMs in the algorithm requiring a number of dirty qubits that is more than what is already provided by the clean qubits.

## Examples
The parameters used to obtain the results in the accompanying paper are given in `params.md`. Some small discrepancies with what is in the paper may occur due to future updates of the code.

## Authorship
### Citation
<!-- ```
@article{casares2022tfermion,
  doi = {10.22331/q-2022-07-20-768},
  url = {https://doi.org/10.22331/q-2022-07-20-768},
  title = {{TF}ermion: {A} non-{C}lifford gate cost assessment library of quantum phase estimation algorithms for quantum chemistry},
  author = {Casares, Pablo A. M. and Campos, Roberto and Martin-Delgado, M. A.},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {6},
  pages = {768},
  month = jul,
  year = {2022}
}
```    -->
### Contributors  
<!-- [Modjtaba Shokrian Zini](https://github.com/mojishoki) (Xanadu Quantum Technologies Inc.).
Xanadu Quantum Technologies Inc. -->
### License
<!-- Apache 2.0 license. -->