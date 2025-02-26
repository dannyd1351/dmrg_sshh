import numpy as np
import scipy.sparse as sp

def operators(operators_type):
    """
    Generates the identity and annihilation operators for a given type of fermionic system.

    Parameters:
    -----------
    operators_type : str
        Specifies the type of system:
        - "spinless" : Spinless fermions (2-dimensional Hilbert space per site).
        - "spinfull" : Spinful fermions (4-dimensional Hilbert space per site).

    Returns:
    --------
    I : scipy.sparse matrix
        Identity matrix for the corresponding Hilbert space.
    Cup : scipy.sparse matrix
        Fermionic annihilation operator for spin-up particles.
    Cdown : scipy.sparse matrix
        Fermionic annihilation operator for spin-down particles (only defined for "spinfull").
    """

    # Case 1: Spinless Fermions (2x2 Hilbert space per site)
    if operators_type == "spinless":
        I = sp.eye(2, format="csr")  # 2x2 identity matrix (sparse format)

        # Fermion annihilation operator (destroys a particle at a site)
        # Basis: |0⟩ (empty), |1⟩ (occupied)
        Cup = sp.csr_matrix((2, 2))  # Initialize a 2x2 sparse matrix with zeros
        Cup[0, 1] = 1  # Converts |1⟩ → |0⟩ (annihilation)

        # Not needed for spinless fermions, but included for consistency
        Cdown = sp.csr_matrix((2, 2))  # Zero matrix (placeholder)

    # Case 2: Spinful Fermions (4x4 Hilbert space per site)
    elif operators_type == "spinfull":
        I = sp.eye(4, format="csr")  # 4x4 identity matrix (sparse format)

        # Spin-up annihilation operator
        # Basis: |0⟩, |↓⟩, |↑⟩, |↑↓⟩
        # Cup removes a spin-up fermion:
        # |↑⟩ → |0⟩  (position [0,2])
        # |↑↓⟩ → |↓⟩ (position [1,3])
        Cup = sp.csr_matrix((4, 4))
        Cup[0, 2] = 1  # |↑⟩ → |0⟩
        Cup[1, 3] = 1  # |↑↓⟩ → |↓⟩

        # Spin-down annihilation operator
        # |↓⟩ → |0⟩  (position [0,1])
        # |↑↓⟩ → |↑⟩ (position [2,3])
        Cdown = sp.csr_matrix((4, 4))
        Cdown[0, 1] = 1  # |↓⟩ → |0⟩
        Cdown[2, 3] = 1  # |↑↓⟩ → |↑⟩

    else:
        raise ValueError("Error: Unknown operators type. Use 'spinless' or 'spinfull'.")

    return I, Cup, Cdown  # Return the identity and annihilation operators
