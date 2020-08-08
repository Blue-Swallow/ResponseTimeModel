# -*- coding: utf-8 -*-
"""
Content: Module for Simulating the Transiend Process of Residence time at Chamber
Author: Ayumu Tsuji @Hokkaido University

Description:
Defined several function to calculate the history of chamber pressure
"""

# %%
import os
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import integrate
import copy
from tqdm import tqdm
from cea_post import Read_datset

#%%
def func_dPc(t, Pc, mox, mf, t_history, Vf_history, **cond):
    Df = cond["Df"]
    Dt = cond["Dt"]
    N = cond["N"]
    Vci = cond["Vci"]
    Af = np.pi*np.power(Df, 2)/4
    At = np.pi*np.power(Dt, 2)/4
    Ru = cond["Ru"]
    func_cstr = cond["func_CSTAR"]
    func_T = cond["func_T"]
    func_M = cond["func_M"]
    eta = cond["eta"]

    Vf = Vf_history[-1]
    if mf<=0.0:
        cstr = Pc*At/mox
        T = cond["T_ox"]
        R = Ru/cond["M_ox"]
    else:
        of = mox/mf
        cstr = eta*func_cstr(of, Pc)
        T = func_T(of, Pc)
        R = Ru/(func_M(of, Pc)*1e-3)
    if t_history[-1] == t:    # in the case of calling this function to calculate "k0"
        t_array_tmp = t_history
        Vf_array_tmp = Vf_history
    else:               # in the case of calling this function to calculate "k1, k2, k3"
        t_array_tmp = np.append(t_history, t)
        Vf_array_tmp = np.append(Vf_history, Vf)
    integ_Vf = integrate.simps(Vf_array_tmp, t_array_tmp)
    dPc = (mox +mf -(Af*Pc*Vf/(R*T) +Pc*At/cstr))/((Vci +Af*integ_Vf)/(R*T))
    return dPc


def exe_EULER(t, mf, Pc, func_mox, t_history, Vf_history, **cond):
    dt = cond["dt"]
    # Pc = val["Pc"]
    Pc_new = Pc + dt*func_dPc(t, Pc, func_mox(t), mf, t_history, Vf_history, **cond)
    return Pc_new