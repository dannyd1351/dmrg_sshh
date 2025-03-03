# Density Matrix Renormalization Group (DMRG) for 1D Fermionic Systems

## Overview
This repository contains a Python implementation of the Density Matrix Renormalization Group (DMRG) algorithm for solving one-dimensional fermionic models, including both spinless and spinful cases. The implementation is designed to be efficient and extensible, allowing for studies of ground state properties, correlation functions (working project), and other observables in strongly correlated electron systems.

## Implemented Hamiltonians
The code currently supports the following fermionic Hamiltonians:

1. **Spinless Fermion Model**
   \[
   H = -t \sum_j \left( c_j^\dagger c_{j+1} + c_{j+1}^\dagger c_j \right) + V \sum_j n_j n_{j+1}
   \]
   where \(t\) is the hopping amplitude, and \(V\) is the nearest-neighbor interaction.

2. **Hubbard Model (Spinful)**
   \[
   H = -t \sum_{j,\sigma} \left( c_{j,\sigma}^\dagger c_{j+1,\sigma} + c_{j+1,\sigma}^\dagger c_{j,\sigma} \right) + U \sum_j n_{j\uparrow} n_{j\downarrow}
   \]
   where \(U\) represents the on-site interaction strength.

3. **SSH Model (Su-Schrieffer-Heeger)**
   \[
   H = - \sum_j \left( t_1 c_j^\dagger c_{j+1} + t_2 c_{j+1}^\dagger c_{j+2} \right)
   \]
   which describes a dimerized hopping chain with alternating hopping parameters \(t_1\) and \(t_2\).

4. **Spinful SSHH Model** (Hubbard model with SSH hopping):
   \[ H = \sum_{i, \sigma} (t_1 c_{i, \sigma}^\dagger c_{i+1, \sigma} + t_2 c_{i+1, \sigma}^\dagger c_{i+2, \sigma} + h.c.) + U \sum_i n_{i,\uparrow} n_{i,\downarrow} \]

## Features
- **Infinite DMRG Warm-up**: Builds the system iteratively to reach a target system size.
- **Finite DMRG Sweeps**: Optimizes the wavefunction by sweeping from left to right and back.
- **Observables**: Computes ground state energy, local densities, correlation functions, and entanglement entropy.
- **Modular Implementation**: Operators, Hamiltonians, and sweep routines are modular for easy extension.

## Installation
To use this repository, clone it and install dependencies:

```bash
$ git clone https://github.com/dannyd1351/dmrg_sshh.git
$ cd dmrg
```

## Usage
The main script `dmrg_main.py` runs the DMRG algorithm based on user-defined parameters provided in `input.txt`.

### Example Input File (`input.txt`)
```plaintext
Model = spinless
m = 20  # Number of states kept per block
m_warm = 10  # Initial warm-up truncation size
N_sweeps = 4  # Number of finite DMRG sweeps
L = 30  # Total system size
Measure = N  # Observable to measure
```

### Running the Code
To execute the DMRG algorithm, run:
```bash
$ python dmrg_main.py
```

The output will be saved in `output.txt`, containing:
- Ground state energy
- Magnetization (for spinful models)
- Other requested observables

## Extending the Code
New Hamiltonians or observables can be added by modifying:
- `hamiltonian.py`: Defines the system's Hamiltonian.
- `operators.py`: Contains the creation, annihilation, and measurement operators.
- `dmrg_main.py`: Controls the main execution flow.

## License
This project is open-source under the MIT License. Contributions are welcome!



