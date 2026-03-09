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
    model.IEN = np.array(FEData['IEN'], dtype=int)
    model.LM = np.zeros((model.nen*model.ndof, model.nel), dtype=int)
    set_LM()

    # element and material data (given at the element)
    model.E     = np.array(FEData['E'])
    model.CArea = np.array(FEData['CArea'])
    model.leng  = np.sqrt(np.power(model.x[model.IEN[:, 1]-1] - 
                                   model.x[model.IEN[:, 0]-1], 2) +
                          np.power(model.y[model.IEN[:, 1]-1] - 
                                   model.y[model.IEN[:, 0]-1], 2))
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
            # insert your code here for 3D
            # ...
            pass # delete or comment this line after your implementation for 3D
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
    
    print("\t2D Truss Params \n")
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
            # insert your code here for 3D
            # ...
            pass # delete or comment this line after your implementation for 3D
        else:
            raise ValueError("The dimension (ndof = {0}) given for the \
                             problem is invalid".format(model.ndof))

        print("{0}\t\t\t{1}".format(e+1, model.stress[e]))


def plot_deformed_truss(scale=None):
    '''
    Plot the original and deformed truss structure.

    Args:
        scale: displacement magnification factor. If None, auto-computed.
    '''
    if model.ndof != 2:
        return

    # auto-scale: make max visible deformation ~10% of structure span
    if scale is None:
        span = max(model.x.max() - model.x.min(), model.y.max() - model.y.min())
        max_disp = max(np.max(np.abs(model.d)), 1e-30)
        scale = 0.1 * span / max_disp

    # compute deformed coordinates
    x_def = model.x.copy()
    y_def = model.y.copy()
    for i in range(model.nnp):
        x_def[i] += scale * model.d[2*i]
        y_def[i] += scale * model.d[2*i + 1]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # 设置中文字体 (macOS: PingFang SC, Linux: SimHei, fallback: sans-serif)
    import platform
    if platform.system() == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti TC']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # plot original truss
    for e in range(model.nel):
        n1, n2 = model.IEN[e, 0] - 1, model.IEN[e, 1] - 1
        ax.plot([model.x[n1], model.x[n2]], [model.y[n1], model.y[n2]],
                'b-', linewidth=1.0, alpha=0.5)

    # plot deformed truss
    for e in range(model.nel):
        n1, n2 = model.IEN[e, 0] - 1, model.IEN[e, 1] - 1
        ax.plot([x_def[n1], x_def[n2]], [y_def[n1], y_def[n2]],
                'r-', linewidth=1.5)

    # plot nodes
    ax.plot(model.x, model.y, 'bo', markersize=5, label='原始结构')
    ax.plot(x_def, y_def, 'rs', markersize=5,
            label='变形结构 (放大系数={:.0f})'.format(scale))

    # node numbers
    for i in range(model.nnp):
        ax.annotate(str(i+1), (model.x[i], model.y[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, color='blue')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('桁架变形图 (位移放大系数 = {:.0f})'.format(scale))
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('deformed_truss.png', dpi=150)
    plt.show()
