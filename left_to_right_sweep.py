import numpy as np
from scipy.sparse.linalg import eigs

def left_to_right_sweep(Model, operators_type, BlockH, BlockHR, I, int_param,
                         Op_block1, Op_block1R, Op_local1, I_block, I_blockR,
                         m, TruncationError, Measure, N):
    
    if operators_type == 'spinless':
        BlockH2 = np.kron(BlockH, I) + int_param * np.kron(Op_block1.T, Op_local1) + int_param * np.kron(Op_block1, Op_local1.T)
        BlockHR2 = np.kron(BlockHR, I) + int_param * np.kron(Op_block1R.T, Op_local1) + int_param * np.kron(Op_block1R, Op_local1.T)
        Op_block12, I_block2, Op_block22 = add_site(Model, Op_block1, I_block)
        Op_block1R2, I_blockR2, Op_block2R2 = add_site(Model, Op_block1R, I_blockR)
    
    elif operators_type == 'spinfull':
        BlockH2 = np.kron(BlockH, I) + int_param * apply_jw_transform(Op_block1.T, Op_local1) + int_param * apply_jw_transform(Op_block1, Op_local1.T)
        BlockHR2 = np.kron(BlockHR, I) + int_param * apply_jw_transform(Op_block1R.T, Op_local1) + int_param * apply_jw_transform(Op_block1R, Op_local1.T)
        Op_block12, I_block2, Op_block22 = add_site(Model, Op_block1, I_block, spinfull=True)
        Op_block1R2, I_blockR2, Op_block2R2 = add_site(Model, Op_block1R, I_blockR, spinfull=True)
    
    else:
        raise ValueError("Model type not yet implemented")
    
    H_super = np.kron(BlockH2, I_blockR2) + np.kron(I_block2, BlockHR2) + int_param * apply_jw_transform(Op_block12.T, Op_block2R2) + int_param * apply_jw_transform(Op_block12, Op_block2R2.T)
    
    Psi, Energy = eigs(H_super, k=1, which='SM')
    
    DimL, DimR = BlockH2.shape[1], BlockHR2.shape[1]
    if m < DimL:
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

        BlockH2 = TL.T @ BlockH2 @ TL
        Op_block12 = TL.T @ Op_block12 @ TL
        Op_block22 = TL.T @ Op_block22 @ TL
        I_block2 = TL.T @ I_block2 @ TL

        BlockHR2 = TR.T @ BlockHR2 @ TR
        Op_block1R2 = TR.T @ Op_block1R2 @ TR
        Op_block2R2 = TR.T @ Op_block2R2 @ TR
        I_blockR2 = TR.T @ I_blockR2 @ TR
    
    if Measure == 'N':
        N2 = N + np.trace((Op_block12.T @ Op_block12) @ RhoL) + np.trace((Op_block1R2.T @ Op_block1R2) @ RhoR)
    else:
        raise ValueError("Measurement type not implemented")
    
    return Psi, Energy, BlockH2, BlockHR2, Op_block12, Op_block1R2, Op_block22, Op_block2R2, I_block2, I_blockR2, TruncationError, N2

def apply_jw_transform(op1, op2):
    sign_factor = np.kron(np.identity(op1.shape[0]), np.identity(op2.shape[1]))
    return sign_factor @ np.kron(op1, op2)

