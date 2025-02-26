import numpy as np
import scipy.sparse as sp

def hamiltonian(model_type):
    """
    Constructs the local Hamiltonian for the given model type.

    The Hamiltonian describes the interactions and hopping terms for various 
    fermionic models. It is initialized as a sparse matrix for computational efficiency.

    Parameters:
    -----------
    model_type : str
        The type of model to construct the Hamiltonian for:
        - "spinless" : Spinless fermion model.
        - "SSH" : Su-Schrieffer-Heeger (SSH) model.
        - "Hubbard" : Hubbard model with electron interactions.
        - "SSHH" : SSH model with Hubbard interactions.

    Returns:
    --------
    Hi : scipy.sparse matrix
        The local Hamiltonian matrix for the specified model.
    """

    # ----------------------- Spinless Fermion Model -----------------------
    if model_type == "spinless":
        """
        Spinless fermion model Hamiltonian.
        The local Hamiltonian is a 2x2 sparse matrix.
        """
        Hi = sp.csr_matrix((2, 2))  # Initialize as a sparse 2x2 matrix

    # ----------------------- SSH Model -----------------------
    elif model_type == "SSH":
        """
        SSH (Su-Schrieffer-Heeger) model.
        The local Hamiltonian is a 2x2 sparse matrix (one spinless fermion per site).
        """
        Hi = sp.csr_matrix((2, 2))  # Initialize as a sparse 2x2 matrix

    # ----------------------- Hubbard Model -----------------------
    elif model_type == "Hubbard":
        """
        Hubbard model Hamiltonian.
        The local Hamiltonian describes the interaction between spin-up and spin-down electrons.
        It is represented as a 4x4 sparse matrix.
        """
        Hi = sp.csr_matrix((4, 4))  # Initialize as a sparse 4x4 matrix

    # ----------------------- SSHH Model (SSH + Hubbard) -----------------------
    elif model_type == "SSHH":
        """
        SSHH (Su-Schrieffer-Heeger-Hubbard) model.
        This combines SSH hopping terms with on-site Hubbard interactions.
        The local Hamiltonian is a 4x4 sparse matrix.
        """
        Hi = sp.csr_matrix((4, 4))  # Initialize as a sparse 4x4 matrix

    else:
        raise ValueError("Error: Hamiltonian type not yet implemented")

    return Hi
