import numpy as np
import jax.numpy as jnp
from jax import jit
from hamiltonian import hamiltonian
from operators import operators
from infinite_dmrg import infinite_dmrg
from left_to_right_sweep import left_to_right_sweep
from right_to_left_sweep import right_to_left_sweep

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            DMRG Main Program (Parallelized with JAX)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load user-defined parameters from input.txt
with open("input.txt", "r") as f:
    params = {}
    for line in f:
        if not line.startswith("#") and line.strip():  # Ignore comments
            key, value = line.split("=")
            params[key.strip()] = value.strip()

Model = params["Model"]
m = int(params["m"])
m_warm = int(params["m_warm"])
N_sweeps = int(params["N_sweeps"])
L = int(params["L"])
Measure = params["Measure"]

t = 1  # Hopping parameter

# Define model-specific parameters
if Model == 'spinless':
    int_param = -t
    operators_type = 'spinless'
elif Model == 'SSH':
    int_param1, int_param2 = 0.5, 1.5
    operators_type = 'spinless'
elif Model == 'Hubbard':
    int_param1 = -t
    operators_type = 'spinfull'
elif Model == 'SSHH':
    int_param1, int_param2 = -1, -1
    operators_type = 'spinfull'
else:
    raise ValueError("Error: Unknown model")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        Initialize Infinite DMRG
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Hloc = hamiltonian(Model)
BlockH = [Hloc]

Op_local1, I, Op_local2 = operators(operators_type)
Op_block1, Op_block2 = [Op_local1], [Op_local2]
I_block = [I]

TruncationError = 0
NIterWarm = L // 2 - 1

# Parallelized matrix-vector multiplication using JAX
@jit
def parallel_matvec(H, psi):
    return jnp.dot(H, psi)

# Infinite DMRG Warm-up
for l in range(NIterWarm):
    Psi, Energy, BlockH_new, Op_block1_new, Op_block2_new, I_block_new, TruncationError = infinite_dmrg(
        Model, l, operators_type, BlockH[-1], I, int_param, Op_block1[-1], Op_local1, I_block[-1], m_warm, TruncationError
    )
    BlockH.append(BlockH_new)
    Op_block1.append(Op_block1_new)
    Op_block2.append(Op_block2_new)
    I_block.append(I_block_new)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      Finite DMRG Sweeps
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_total_prev = 0

for s in range(N_sweeps):
    left, right = len(BlockH) - 1, len(BlockH) - 3

    while right > 0:
        Psi, Energy, *_ = left_to_right_sweep(
            Model, operators_type, BlockH[left], BlockH[right], I, int_param,
            Op_block1[left], Op_block1[right], Op_local1, I_block[left], I_block[right],
            m, TruncationError, Measure, N_total_prev
        )
        left += 1
        right -= 1

    left -= 1
    right += 1
    while left > 0:
        Psi, Energy, *_ = right_to_left_sweep(
            Model, operators_type, BlockH[right], BlockH[left], I, int_param,
            Op_block1[right], Op_block1[left], Op_local1, I_block[right], I_block[left],
            m, TruncationError
        )
        left -= 1
        right += 1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         Save Output
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output_filename = "output.txt"
with open(output_filename, "w") as f:
    f.write("DMRG Calculation Summary\n")
    f.write("--------------------------------------\n")
    f.write(f"Model: {Model}\n")
    f.write(f"Total Sites: {L}\n")
    f.write(f"Number of States (m): {m}\n")
    f.write(f"Number of Sweeps: {N_sweeps}\n")
    f.write(f"Observable: {Measure}\n")
    f.write("--------------------------------------\n")
    f.write(f"Ground state energy (in units of t): {Energy.flatten()[0]:.6f}\n")
print(f"Calculation completed. Results saved in {output_filename}")
