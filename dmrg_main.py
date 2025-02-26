import numpy as np
from hamiltonian import hamiltonian
from operators import operators
from infinite_dmrg import infinite_dmrg
from left_to_right_sweep import left_to_right_sweep
from right_to_left_sweep import right_to_left_sweep

# Function to read input parameters from a file
def read_input(filename="input.txt"):
    params = {}
    with open(filename, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=")
                params[key.strip()] = value.strip()
    return params

# Read parameters from input.txt
params = read_input()

# Assign parameters
Model = params.get("Model", "spinless")
m = int(params.get("m", 10))
m_warm = int(params.get("m_warm", 10))
N_sweeps = int(params.get("N_sweeps", 4))
L = int(params.get("L", 4))
Measure = params.get("Measure", "N")

# Define model parameters
t = 1  # Default hopping parameter

if Model == 'spinless':
    int_param = -t
    operators_type = 'spinless'
elif Model == 'SSH':
    int_param1 = 0.5
    int_param2 = 1.5
    operators_type = 'spinless'
elif Model == 'Hubbard':
    int_param1 = -t
    operators_type = 'spinfull'
elif Model == 'SSHH':
    int_param1 = -1
    int_param2 = -1
    operators_type = 'spinfull'
else:
    raise ValueError("Error: Unknown model")

# Initialize DMRG
Hloc = hamiltonian(Model)
BlockH = [Hloc]

Op_local1, I, Op_local2 = operators(operators_type)
Op_block1 = [Op_local1]
I_block = [I]
Op_block2 = [Op_local2]

TruncationError = 0
NIterWarm = L // 2 - 1

# Infinite DMRG Warm-up
for l in range(NIterWarm):
    Psi, Energy, BlockH_new, Op_block1_new, Op_block2_new, I_block_new, TruncationError = infinite_dmrg(
        Model, l, operators_type, BlockH[-1], I, int_param, Op_block1[-1], Op_local1, I_block[-1], m_warm, TruncationError
    )
    BlockH.append(BlockH_new)
    Op_block1.append(Op_block1_new)
    Op_block2.append(Op_block2_new)
    I_block.append(I_block_new)

# Finite DMRG Sweeps
N_total_prev = 0
for s in range(N_sweeps):
    left = len(BlockH) - 1
    right = len(BlockH) - 3

    while right > 0:
        (Psi, Energy, BlockH[left + 1], BlockH[right + 1], Op_block1[left + 1], Op_block1[right + 1],
         Op_block2[left + 1], Op_block2[right + 1], I_block[left + 1], I_block[right + 1],
         TruncationError, N_total) = left_to_right_sweep(
            Model, operators_type, BlockH[left], BlockH[right], I, int_param,
            Op_block1[left], Op_block1[right], Op_local1, I_block[left], I_block[right],
            m, TruncationError, Measure, N_total_prev
        )
        left += 1
        right -= 1

    left -= 1
    right += 1
    while left > 0:
        (Psi, Energy, BlockH[right + 1], BlockH[left + 1], Op_block1[right + 1], Op_block1[left + 1],
         Op_block2[right + 1], Op_block2[left + 1], I_block[right + 1], I_block[left + 1],
         TruncationError) = right_to_left_sweep(
            Model, operators_type, BlockH[right], BlockH[left], I, int_param,
            Op_block1[right], Op_block1[left], Op_local1, I_block[right], I_block[left],
            m, TruncationError
        )
        left -= 1
        right += 1

    left += 1
    right -= 1
    while left <= right:
        (Psi, Energy, BlockH[left + 1], BlockH[right + 1], Op_block1[left + 1], Op_block1[right + 1],
         Op_block2[left + 1], Op_block2[right + 1], I_block[left + 1], I_block[right + 1],
         TruncationError, N_total) = left_to_right_sweep(
            Model, operators_type, BlockH[left], BlockH[right], I, int_param,
            Op_block1[left], Op_block1[right], Op_local1, I_block[left], I_block[right],
            m, TruncationError, Measure, N_total_prev
        )
        left += 1
        right -= 1

# Write the summary and result to an output file
output_filename = "output.txt"

with open(output_filename, "w") as f:
    f.write("DMRG Calculation Summary\n")
    f.write("========================\n")
    f.write(f"Model: {Model}\n")
    f.write(f"Number of sites (L): {L}\n")
    f.write(f"Number of states kept (m): {m}\n")
    f.write(f"Warm-up states (m_warm): {m_warm}\n")
    f.write(f"Number of sweeps: {N_sweeps}\n")
    f.write(f"Measured observable: {Measure}\n")
    f.write("\n")
    f.write(f"Ground state energy: {Energy.flatten()[0]:.6f} t\n")
    f.write(f"Truncation error: {TruncationError:.2e}\n")

print(f"Calculation completed. Results saved in {output_filename}")

