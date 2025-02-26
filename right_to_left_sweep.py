import numpy as np
from scipy.sparse.linalg import eigs

def right_to_left_sweep(Model, operators_type, BlockH, BlockHL, I, int_param,
                         Op_block1, Op_block1L, Op_local1, I_block, I_blockL,
                         m, TruncationError):
    
    if operators_type == 'spinless':
        BlockH2 = np.kron(BlockH, I) + int_param * np.kron(Op_block1.T, Op_local1) + int_param * np.kron(Op_block1, Op_local1.T)
        BlockHL2 = np.kron(BlockHL, I) + int_param * np.kron(Op_block1L.T, Op_local1) + int_param * np.kron(Op_block1L, Op_local1.T)
        Op_block12, I_block2, Op_block22 = add_site(Model, Op_block1, I_block)
        Op_block1L2, I_blockL2, Op_block2L2 = add_site(Model, Op_block1L, I_blockL)
    
    elif operators_type == 'spinfull':
        BlockH2 = np.kron(BlockH, I) + int_param * apply_jw_transform(Op_block1.T, Op_local1) + int_param * apply_jw_transform(Op_block1, Op_local1.T)
        BlockHL2 = np.kron(BlockHL, I) + int_param * apply_jw_transform(Op_block1L.T, Op_local1) + int_param * apply_jw_transform(Op_block1L, Op_local1.T)
        Op_block12, I_block2, Op_block22 = add_site(Model, Op_block1, I_block, spinfull=True)
        Op_block1L2, I_blockL2, Op_block2L2 = add_site(Model, Op_block1L, I_blockL, spinfull=True)
    
    else:
        raise ValueError("Model type not yet implemented")
    
    H_super = np.kron(BlockHL2, I_block2) + np.kron(I_blockL2, BlockH2) + int_param * apply_jw_transform(Op_block1L2.T, Op_block22) + int_param * apply_jw_transform(Op_block1L2, Op_block22.T)
    
    Psi, Energy = eigs(H_super, k=1, which='SM')
    
    DimL, DimR = BlockHL2.shape[1], BlockH2.shape[1]
    if m < DimR:
        PsiMatrix = Psi.reshape(DimR, DimL)
        RhoL = PsiMatrix.T @ PsiMatrix
        RhoR = PsiMatrix @ PsiMatrix.T

        VL, DL = np.linalg.eigh(RhoL)
        IndexL = np.argsort(DL)[::-1]
        VL = VL[:, IndexL]

        VR, DR = np.linalg.eigh(RhoR)
        IndexR = np.argsort(DR)[::-1]
        VR = VR[:, IndexR]

        NKeepL, NKeepR = min(DL.shape[0], m), min(DR.shape[0], m)
        TL, TR = VL[:, :NKeepL], VR[:, :NKeepR]
        TruncationError += (1 - np.sum(DL[:NKeepL])) + (1 - np.sum(DR[:NKeepR]))

        BlockH2 = TR.T @ BlockH2 @ TR
        Op_block12 = TR.T @ Op_block12 @ TR
        Op_block22 = TR.T @ Op_block22 @ TR
        I_block2 = TR.T @ I_block2 @ TR

        BlockHL2 = TL.T @ BlockHL2 @ TL
        Op_block1L2 = TL.T @ Op_block1L2 @ TL
        Op_block2L2 = TL.T @ Op_block2L2 @ TL
        I_blockL2 = TL.T @ I_blockL2 @ TL
    
    return Psi, Energy, BlockH2, BlockHL2, Op_block12, Op_block1L2, Op_block22, Op_block2L2, I_block2, I_blockL2, TruncationError

def apply_jw_transform(op1, op2):
    sign_factor = np.kron(np.identity(op1.shape[0]), np.identity(op2.shape[1]))
    return sign_factor @ np.kron(op1, op2)

