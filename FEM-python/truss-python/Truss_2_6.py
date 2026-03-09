#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
习题 2.6 - 平面桁架问题
比较缩减法 (reduction method) 和罚函数法 (penalty method) 施加位移边界条件的结果,
并绘制变形结构图。
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 确保能导入 truss-python 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import FEData as model
from TrussElem import TrussElem
from PrePost import create_model_json, print_stress, plot_deformed_truss
from utitls import assembly, solvedr, solvep


def compute_internal_forces():
    """计算各杆件内力 N = stress * CArea"""
    forces = np.zeros(model.nel)
    for e in range(model.nel):
        de = model.d[model.LM[:, e]]
        const = model.E[e] / model.leng[e]
        IENe = model.IEN[e] - 1
        xe = model.x[IENe]
        ye = model.y[IENe]
        s = (ye[1] - ye[0]) / model.leng[e]
        c = (xe[1] - xe[0]) / model.leng[e]
        stress = const * (np.array([-c, -s, c, s]) @ de)
        forces[e] = stress * model.CArea[e]
    return forces


def element_description(e):
    """返回单元描述: 节点号和类型"""
    n1, n2 = model.IEN[e, 0], model.IEN[e, 1]
    return "({:2d}-{:2d})".format(n1, n2)


def main():
    DataFile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'truss_2_6.json')

    # ========== 创建模型 & 组装 ==========
    create_model_json(DataFile)
    plt.close('all')

    for e in range(model.nel):
        ke = TrussElem(e)
        assembly(e, ke)

    # 保存原始状态
    K_orig = model.K.copy()
    d_orig = model.d.copy()
    f_orig = model.f.copy()

    # ========== 方法1: 缩减法 ==========
    print("\n" + "="*60)
    print("         缩减法 (Reduction Method)")
    print("="*60)
    f_E_r = solvedr()
    print_stress()
    d_reduction = model.d.copy()
    stress_reduction = model.stress.copy()
    N_reduction = compute_internal_forces()

    # ========== 重置 & 方法2: 罚函数法 ==========
    model.K = K_orig.copy()
    model.d = d_orig.copy()
    model.f = f_orig.copy()
    model.stress = np.zeros(model.nel)

    print("\n" + "="*60)
    print("         罚函数法 (Penalty Method, α = 1e7)")
    print("="*60)
    f_E_p = solvep(alpha=1e7)
    print_stress()
    d_penalty = model.d.copy()
    stress_penalty = model.stress.copy()
    N_penalty = compute_internal_forces()

    # ========== 比较结果 ==========
    print("\n" + "="*60)
    print("         结果比较")
    print("="*60)

    # 节点位移比较
    print("\n--- 节点位移对比 ---")
    print("{:>4s}  {:>14s}  {:>14s}  {:>14s}  {:>14s}  {:>12s}  {:>12s}".format(
        "节点", "ux (缩减法)", "uy (缩减法)", "ux (罚函数)", "uy (罚函数)",
        "Δux", "Δuy"))
    print("-" * 100)
    for i in range(model.nnp):
        ux_r = d_reduction[2*i]
        uy_r = d_reduction[2*i+1]
        ux_p = d_penalty[2*i]
        uy_p = d_penalty[2*i+1]
        print("{:4d}  {:14.6e}  {:14.6e}  {:14.6e}  {:14.6e}  {:12.2e}  {:12.2e}".format(
            i+1, ux_r, uy_r, ux_p, uy_p, abs(ux_r-ux_p), abs(uy_r-uy_p)))

    # 内力比较
    print("\n--- 杆件内力 N (N) 对比 ---")
    print("{:>4s}  {:>8s}  {:>14s}  {:>14s}  {:>12s}".format(
        "单元", "节点", "N (缩减法)", "N (罚函数)", "差值 ΔN"))
    print("-" * 65)
    for e in range(model.nel):
        desc = element_description(e)
        diff = abs(N_reduction[e] - N_penalty[e])
        print("{:4d}  {:>8s}  {:14.4f}  {:14.4f}  {:12.2e}".format(
            e+1, desc, N_reduction[e], N_penalty[e], diff))

    # 支座反力比较
    print("\n--- 支座反力 (N) 对比 ---")
    bc_labels = []
    for dof in model.edof:
        node = dof // 2 + 1
        direction = 'x' if dof % 2 == 0 else 'y'
        bc_labels.append("Node {} R{}".format(node, direction))
    print("{:>15s}  {:>14s}  {:>14s}  {:>12s}".format(
        "反力", "缩减法", "罚函数法", "差值"))
    print("-" * 60)
    for i in range(len(model.edof)):
        diff = abs(f_E_r[i] - f_E_p[i])
        print("{:>15s}  {:14.4f}  {:14.4f}  {:12.2e}".format(
            bc_labels[i], f_E_r[i], f_E_p[i], diff))

    # 汇总
    print("\n--- 误差汇总 ---")
    print("位移最大绝对误差:  {:.6e}".format(np.max(np.abs(d_reduction - d_penalty))))
    print("位移最大相对误差:  {:.6e}".format(
        np.max(np.abs(d_reduction - d_penalty)[np.abs(d_reduction) > 1e-10] /
               np.abs(d_reduction[np.abs(d_reduction) > 1e-10]))))
    print("内力最大绝对误差:  {:.6e} N".format(np.max(np.abs(N_reduction - N_penalty))))
    print("反力最大绝对误差:  {:.6e} N".format(np.max(np.abs(f_E_r - f_E_p))))

    # ========== 绘制变形图 (使用缩减法结果) ==========
    model.d = d_reduction
    plot_deformed_truss(scale=5000)


if __name__ == "__main__":
    main()
