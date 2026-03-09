#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides utilities used by FE analysis.
  1. assembly: Global stiffness matrix assembly.
  2. solvedr: Solving the stiffness equations by the reduction approach.
  3. solvep:  Solving the stiffness equations by the penalty method.

Created on Sat May 9 17:39:00 2020

@author: thurcni@163.com, xzhang@tsinghua.edu.cn
"""

import numpy as np
import FEData as model


def assembly(e, ke):
    """
    Assemble element stiffness matrix.
    
    Args:
        e   : (int) Element number
        ke  : (numpy(nen*ndof,nen*ndof)) element stiffness matrix
    """
    model.K[np.ix_(model.LM[:,e], model.LM[:,e])] += ke

def solvedr():
    """
    Partition and solve the system of equations using the reduction method.
    Supports arbitrary essential DOF indices via model.edof.
        
    Returns:
        f_E : (numpy.array(nd,)) Reaction force vector at essential DOFs
    """
    neq = model.neq
    edof = model.edof                                  # 0-indexed essential DOFs
    fdof = np.setdiff1d(np.arange(neq), edof)           # free DOFs

    K_E  = model.K[np.ix_(edof, edof)]
    K_F  = model.K[np.ix_(fdof, fdof)]
    K_EF = model.K[np.ix_(edof, fdof)]
    f_F  = model.f[fdof].flatten()
    d_E  = model.d[edof].flatten()

    # solve for d_F
    d_F = np.linalg.solve(K_F, f_F - K_EF.T @ d_E)

    # reconstruct the global displacement d
    d = np.zeros(neq)
    d[edof] = d_E
    d[fdof] = d_F
    model.d = d

    # compute the reaction f_E
    f_E = K_E @ d_E + K_EF @ d_F

    # write to the workspace
    print('\nsolution d');  print(model.d)
    print('\nreaction f =', f_E)

    return f_E


def solvep(alpha=1e7):
    """
    Solve the system of equations using the penalty method.

    Args:
        alpha : (float) Penalty ratio. The penalty parameter 
                beta = alpha * mean(|K_ii|), where mean(|K_ii|) is the 
                average of absolute values of diagonal entries of K.

    Returns:
        f_E : (numpy.array(nd,)) Reaction force vector at essential DOFs
    """
    neq  = model.neq
    edof = model.edof
    e_bc = model.e_bc

    # penalty parameter: beta = alpha * average of |K_ii|
    K_diag_avg = np.mean(np.abs(np.diag(model.K)))
    beta = alpha * K_diag_avg
    print('\nPenalty parameter: beta = {:.6e} (alpha = {:.0e}, avg|K_ii| = {:.6e})'.format(
        beta, alpha, K_diag_avg))

    # apply penalty terms (work on copies so original K, f are preserved)
    K_pen = model.K.copy()
    f_pen = model.f.flatten().copy()

    for i, dof in enumerate(edof):
        K_pen[dof, dof] += beta
        f_pen[dof]      += beta * e_bc[i]

    # solve
    model.d = np.linalg.solve(K_pen, f_pen)

    # compute reactions: R_i = beta * (d_bar_i - d_i)
    f_E = np.zeros(len(edof))
    for i, dof in enumerate(edof):
        f_E[i] = beta * (e_bc[i] - model.d[dof])

    # write to the workspace
    print('\nsolution d');  print(model.d)
    print('\nreaction f =', f_E)

    return f_E