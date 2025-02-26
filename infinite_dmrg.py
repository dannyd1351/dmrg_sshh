import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigs

def jordan_wigner_transform(L):
    """Constructs Jordan-Wigner string operators for spinfull fermions."""
    JW_string = [identity(2**i) for i in range(L)]
    for i in range(1, L):
        JW_string[i] = kron(JW_string[i-1], csr_matrix(np.array([[1, 0], [0, -1]])), format='csr')
    return JW_string

def infinite_dmrg(model, l, operators_type, BlockH, I, int_param, Op_block1, Op_local1, I_block, m, TruncationError):
    """
    Implements the infinite DMRG algorithm to grow the system iteratively.
    """
    if operators_type == 'spinless':
        # Construct the enlarged block Hamiltonian for spinless fermions
        BlockH2 = kron(BlockH, I) + int_param * kron(Op_block1.T, Op_local1) + int_param * kron(Op_block1, Op_local1.T)
        Op_block12 = kron(Op_block1, I)
        Op_block22 = kron(I, Op_block1)
        I_block2 = kron(I, I)
    
    elif operators_type == 'spinfull':
        # Jordan-Wigner transformation to enforce fermionic statistics
        JW_string = jordan_wigner_transform(l+1)
        C_up = kron(Op_block1, I) @ JW_string[l]
        C_down = kron(I, Op_block1) @ JW_string[l]
        
        # Construct the enlarged Hamiltonian
        BlockH2 = kron(BlockH, I) + int_param * (kron(C_up.T, C_up) + kron(C_up, C_up.T) +
                                                 kron(C_down.T, C_down) + kron(C_down, C_down.T))
        Op_block12 = C_up
        Op_block22 = C_down
        I_block2 = kron(I, I)
    
    else:
        raise ValueError('Unknown operators type')
    
    # Construct the full superblock Hamiltonian
    H_super = kron(BlockH2, I_block2) + kron(I_block2, BlockH2) + int_param * (kron(Op_block12.T, Op_block22) +
                                                                              kron(Op_block12, Op_block22.T))
    H_super = 0.5 * (H_super + H_super.T)  # Ensure symmetry
    
    # Diagonalize the superblock Hamiltonian
    Psi, Energy = eigs(H_super, k=1, which='SM')
    
    # Form the reduced density matrix
    Dim = int(np.sqrt(Psi.shape[0]))
    if m < Dim:
        PsiMatrix = Psi.reshape(Dim, Dim)
        Rho = PsiMatrix.T @ PsiMatrix
        
        # Diagonalize the density matrix
        D, V = np.linalg.eigh(Rho)
        D_sorted_indices = np.argsort(D)[::-1]  # Sort eigenvalues in descending order
        V = V[:, D_sorted_indices]
        
        # Construct the truncation operator
        NKeep = min(len(D), m)
        T = V[:, :NKeep]
        TruncationError += 1 - np.sum(D[:NKeep])
        
        # Transform the block operators into the truncated basis
        BlockH2 = T.T @ BlockH2 @ T
        Op_block12 = T.T @ Op_block12 @ T
        Op_block22 = T.T @ Op_block22 @ T
        I_block2 = T.T @ I_block2 @ T

    """
   Returns:
    Psi : numpy.ndarray
        The ground state wavefunction of the system.
    Energy : float
        The ground state energy.
    BlockH2 : scipy.sparse matrix
        The new Hamiltonian after adding a site.
    Op_block12 : scipy.sparse matrix
        Updated block operator after adding a site.
    Op_block22 : scipy.sparse matrix
        Second block operator after adding a site.
    I_block2 : scipy.sparse matrix
        Updated identity matrix for the new block.
    TruncationError : float
        Updated truncation error after this step.
    """

    return Psi, Energy, BlockH2, Op_block12, Op_block22, I_block2, TruncationError

