# Parameters of the PP runs
To optimize depth for all four materials, we chose the default values `n_tof = 500`, `p_th = 0.75` and used the chemical accuracy for the target error `error = 1.5e-3`. The value of `n_dirty` for each material (which has to be specified inside the code) was determined by first running the similar algorithm for the AE case, as discussed in the paper, using the parameters further below. To optimize cost, the only change is `n_tof = 1`.

# Parameters of the AE runs
The circuit design of the AE algorithm (and as a result its cost/depth) is sensitive to the amplitude amplification threshold `p_th`. As mentioned in the paper, we selected the one that offered the lowest cost/depth. Here are the values of `p_th` for each material and `N` (note this value does not depend on `n_tof`). The target error is always the chemical accuracy.

For `dilithium_iron_silicate = Li2FeSiO4`:
```
N = 1e3, p_th = 0.92
N = 1e4, p_th = 0.8
N = 1e5, p_th = 0.8
N = N^AE = 5.46e7, p_th = 0.78
```
For `Li_rich_manganese_oxyfluoride = Li0.5MnO2F`:
```
N = 1e3, p_th = 0.93
N = 1e4, p_th = 0.8
N = 1e5, p_th = 0.81
N = N^AE = 6.4e8, p_th = 0.84
```
For `LLNMO = Li0.75[Li0.17Mn0.58Ni0.25]O2`:
```
N = 1e3, p_th = 0.95
N = 1e4, p_th = 0.95
N = 1e5, p_th = 0.88
N = N^AE = 8.7e8, p_th = 0.88
```
For `Li_rich_manganese_oxide= Li0.5MnO3`:
```
N = 1e3, p_th = 0.95
N = 1e4, p_th = 0.84
N = 1e5, p_th = 0.82
N = N^AE = 5.8e8, p_th = 0.82
```
