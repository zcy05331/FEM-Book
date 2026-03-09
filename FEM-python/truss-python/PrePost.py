#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provide methods to setup LM matrices, create FE model for a truss from a json 
file, to plot the truss, to calculate and print stresses of every element.

Created on Sat May 9 15:43:00 2020

@author: thurcni@163.com, xzhang@tsinghua.edu.cn
"""

import FEData as model
import numpy as np
import json
import matplotlib.pyplot as plt


def create_model_json(DataFile):
    """ 
    Initialize the FEM model from file DataFile (in json format)
    """

    # input data from json file
    with open(DataFile) as f_obj:
        FEData = json.load(f_obj)
    
    model.Title= FEData['Title']
    model.nsd  = FEData['nsd']
    model.ndof = FEData['ndof']
    model.nnp  = FEData['nnp']
    model.nel  = FEData['nel']
    model.nen  = FEData['nen']    
    model.neq  = model.ndof*model.nnp
    model.nd   = FEData['nd']

    # initialize K, d and f 
    model.f = np.zeros((model.neq,1))            
    model.d = np.zeros((model.neq,1))        
    model.K = np.zeros((model.neq,model.neq))

    # define the mesh
    model.x = np.array(FEData['x'])
    model.y = np.array(FEData['y'])
    model.z = np.array(FEData.get('z', [0.0]*model.nnp))
    model.IEN = np.array(FEData['IEN'], dtype=int)
    model.LM = np.zeros((model.nen*model.ndof, model.nel), dtype=int)
    set_LM()

    # element and material data (given at the element)
    model.E     = np.array(FEData['E'])
    model.CArea = np.array(FEData['CArea'])
    model.leng  = np.sqrt(np.power(model.x[model.IEN[:, 1]-1] - 
                                   model.x[model.IEN[:, 0]-1], 2) +
                          np.power(model.y[model.IEN[:, 1]-1] - 
                                   model.y[model.IEN[:, 0]-1], 2) +
                          np.power(model.z[model.IEN[:, 1]-1] - 
                                   model.z[model.IEN[:, 0]-1], 2))
    model.stress= np.zeros((model.nel,))

    # prescribed forces
    fdof = FEData['fdof']
    force= FEData['force']
    for ind, value in enumerate(fdof):
        model.f[value-1][0] = force[ind]

    # essential boundary conditions
    if 'edof' in FEData:
        # new format: explicit DOF indices (1-indexed in JSON)
        model.edof = np.array(FEData['edof'], dtype=int) - 1  # convert to 0-indexed
        model.e_bc = np.array(FEData['e_bc'])
    else:
        # old format: first nd DOFs are essential BCs
        model.edof = np.arange(model.nd, dtype=int)
        model.e_bc = np.array(FEData['d'][:model.nd])

    # set prescribed displacements in d
    for i, dof in enumerate(model.edof):
        model.d[dof][0] = model.e_bc[i]

    # output plots
    model.plot_truss= FEData['plot_truss']
    model.plot_node = FEData['plot_node']
    model.plot_tex  = FEData['plot_tex']
    plottruss()


def set_LM():
    '''
    set up Location Matrix
    '''
    for e in range(model.nel):
        for j in range(model.nen):
            for m in range(model.ndof):
                ind = j*model.ndof + m
                model.LM[ind, e] = model.ndof*(model.IEN[e, j] - 1) + m


def plottruss():
    '''
    plot the truss
    '''
    if model.plot_truss == "yes":
        if model.ndof == 1:
            for i in range(model.nel):
                XX = np.array([model.x[model.IEN[i, 0]-1], 
                               model.x[model.IEN[i, 1]-1]])
                YY = np.array([0.0, 0.0])
                plt.plot(XX, YY, "blue")

                if model.plot_node == "yes":
                    plt.text(XX[0], YY[0], str(model.IEN[i, 0]))
                    plt.text(XX[1], YY[1], str(model.IEN[i, 1]))
        elif model.ndof == 2:
            for i in range(model.nel):
                XX = np.array([model.x[model.IEN[i, 0]-1], 
                               model.x[model.IEN[i, 1]-1]])
                YY = np.array([model.y[model.IEN[i, 0]-1], 
                               model.y[model.IEN[i, 1]-1]])
                plt.plot(XX, YY, "blue")

                if model.plot_node == "yes":
                    plt.text(XX[0], YY[0], str(model.IEN[i, 0]))
                    plt.text(XX[1], YY[1], str(model.IEN[i, 1]))
        elif model.ndof == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig3d = plt.figure()
            ax3d = fig3d.add_subplot(111, projection='3d')
            for i in range(model.nel):
                XX = [model.x[model.IEN[i, 0]-1], model.x[model.IEN[i, 1]-1]]
                YY = [model.y[model.IEN[i, 0]-1], model.y[model.IEN[i, 1]-1]]
                ZZ = [model.z[model.IEN[i, 0]-1], model.z[model.IEN[i, 1]-1]]
                ax3d.plot(XX, YY, ZZ, "blue")
                if model.plot_node == "yes":
                    ax3d.text(XX[0], YY[0], ZZ[0], str(model.IEN[i, 0]))
                    ax3d.text(XX[1], YY[1], ZZ[1], str(model.IEN[i, 1]))
            ax3d.set_title("3D Truss Plot")
            ax3d.set_xlabel(r"$x$")
            ax3d.set_ylabel(r"$y$")
            ax3d.set_zlabel(r"$z$")
            plt.show()
        else:
            raise ValueError("The dimension (ndof = {0}) given for the \
                             plottruss is invalid".format(model.ndof))
        
        plt.title("Truss Plot")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        # Convert matplotlib figures into PGFPlots figures stored in a Tikz file, 
        # which can be added into your LaTex source code by "\input{fe_plot.tex}"
        if model.plot_tex == "yes":
            import tikzplotlib
            tikzplotlib.clean_figure()
            tikzplotlib.save("fe_plot.tex")
    
    print("\t{}D Truss Params \n".format(model.nsd))
    print(model.Title + "\n")
    print("No. of Elements  {0}".format(model.nel))
    print("No. of Nodes     {0}".format(model.nnp))
    print("No. of Equations {0}".format(model.neq))


def print_stress():
    '''
    Calculate and print stresses of every element
    '''

    # prints the element number and corresponding stresses
    print("Element\t\t\tStress")
    # Compute stress for each element
    for e in range(model.nel):
        de = model.d[model.LM[:, e]]  # nodal displacements for each element
        const = model.E[e]/model.leng[e]

        if model.ndof == 1:
            model.stress[e] = const*(np.array([-1, 1])@de)
        elif model.ndof == 2:
            IENe = model.IEN[e] - 1
            xe = model.x[IENe]
            ye = model.y[IENe]
            s = (ye[1] - ye[0])/model.leng[e]
            c = (xe[1] - xe[0])/model.leng[e]
            model.stress[e] = const*(np.array([-c, -s, c, s])@de)
        elif model.ndof == 3:
            IENe = model.IEN[e] - 1
            l = (model.x[IENe[1]] - model.x[IENe[0]]) / model.leng[e]
            m = (model.y[IENe[1]] - model.y[IENe[0]]) / model.leng[e]
            n = (model.z[IENe[1]] - model.z[IENe[0]]) / model.leng[e]
            model.stress[e] = const * (np.array([-l, -m, -n, l, m, n]) @ de)
        else:
            raise ValueError("The dimension (ndof = {0}) given for the \
                             problem is invalid".format(model.ndof))

        print("{0}\t\t\t{1}".format(e+1, model.stress[e]))


def plot_deformed_truss(scale=None):
    '''
    Plot the original and deformed truss structure.
    Supports ndof=2 (2D) and ndof=3 (3D, including planar 3D with z=0).

    Args:
        scale: displacement magnification factor. If None, auto-computed.
    '''
    if model.ndof == 1:
        return

    # detect planar 3D (all z coords and uz displacements essentially zero)
    is_planar_3d = (model.ndof == 3 and
                    np.allclose(model.z, 0.0, atol=1e-10) and
                    np.allclose(model.d[2::3], 0.0, atol=1e-14))

    # auto-scale: make max visible deformation ~10% of structure span
    if scale is None:
        span = max(model.x.max() - model.x.min(), model.y.max() - model.y.min())
        if model.ndof == 3 and not is_planar_3d:
            span = max(span, model.z.max() - model.z.min())
        max_disp = max(np.max(np.abs(model.d)), 1e-30)
        scale = 0.1 * span / max_disp

    # ---------- 2D plot (ndof=2, or planar 3D) ----------
    if model.ndof == 2 or is_planar_3d:
        ndof = model.ndof
        x_def = model.x.copy()
        y_def = model.y.copy()
        for i in range(model.nnp):
            x_def[i] += scale * model.d[ndof*i]
            y_def[i] += scale * model.d[ndof*i + 1]

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))

        import platform
        if platform.system() == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti TC']
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False

        for e in range(model.nel):
            n1, n2 = model.IEN[e, 0] - 1, model.IEN[e, 1] - 1
            ax.plot([model.x[n1], model.x[n2]], [model.y[n1], model.y[n2]],
                    'b-', linewidth=1.0, alpha=0.5)
        for e in range(model.nel):
            n1, n2 = model.IEN[e, 0] - 1, model.IEN[e, 1] - 1
            ax.plot([x_def[n1], x_def[n2]], [y_def[n1], y_def[n2]],
                    'r-', linewidth=1.5)

        ax.plot(model.x, model.y, 'bo', markersize=5, label='原始结构')
        suffix = ' (3D程序)' if is_planar_3d else ''
        ax.plot(x_def, y_def, 'rs', markersize=5,
                label='变形结构 (放大系数={:.0f}){}'.format(scale, suffix))
        for i in range(model.nnp):
            ax.annotate(str(i+1), (model.x[i], model.y[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8, color='blue')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('桁架变形图 (位移放大系数 = {:.0f}){}'.format(scale, suffix))
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # ---------- 3D plot ----------
    else:
        from mpl_toolkits.mplot3d import Axes3D
        x_def = model.x.copy()
        y_def = model.y.copy()
        z_def = model.z.copy()
        for i in range(model.nnp):
            x_def[i] += scale * model.d[3*i]
            y_def[i] += scale * model.d[3*i + 1]
            z_def[i] += scale * model.d[3*i + 2]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for e in range(model.nel):
            n1, n2 = model.IEN[e, 0] - 1, model.IEN[e, 1] - 1
            ax.plot([model.x[n1], model.x[n2]], [model.y[n1], model.y[n2]],
                    [model.z[n1], model.z[n2]], 'b-', lw=1.0, alpha=0.5)
        for e in range(model.nel):
            n1, n2 = model.IEN[e, 0] - 1, model.IEN[e, 1] - 1
            ax.plot([x_def[n1], x_def[n2]], [y_def[n1], y_def[n2]],
                    [z_def[n1], z_def[n2]], 'r-', lw=1.5)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.set_title('3D桁架变形图 (放大系数 = {:.0f})'.format(scale))

    plt.tight_layout()
    plt.savefig('deformed_truss.png', dpi=150)
    plt.show()
