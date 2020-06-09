# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:56:46 2018

@author: TJ
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
import mod_shape

#%%
"""##########################
# Define unchange parameter #  
##########################"""
# Df = 38e-3 #　[m] fuel outer diamter
# a = 0.981 #　[-] fuel filling rate
# Dt = 6.2e-3 # [m] nozzle thorat diamter
# rho_f = 1191 # [kg/m^3] fuel density
# R_ox = 8.3144598/32e-3 #　[J/kg/K] oxidizer gas constant
# T_ox = 300 #　[K] oxidizer temperature

# %%
"""#####################
# Define initial value #  
#####################"""
# Lci = 20.0e-3 #　[m] initial chamber length
# Vci = np.pi*np.power(Df,2)/4 * Lci #　[m^3] initial chamber volume

# %%
"""#################
# Define functions #  
#################"""
def func_Vf(Pc,Vox):
    """ Calculate fuel reguression rate 
    
    Parameter
    ------------
    Pc: float
        chamber pressure [Pa]
    Vox: float
        oxidizer port velocity [m/s]
    
    Return
    ------------
    Vf: float
        fuel regression rate
    """
    n = 1.0
    C1  = 1.39e-7
    C2 = 1.61e-9
    Vf = (C1/Vox + C2)*np.power(Pc,n)
    return(Vf)

def func_Vox(Pc, mox, **kwargs):
    """ Calculate oxidizer port velocity
    
    Paramter
    -----------
    Pc: float
        chamber pressure [Pa]
    mox: float
        oxidizer mass flow rate [kg/s]
        
    Return
    -----------
    Vox: float
        oxidizer port velocity [m/s]
    """
    R_ox = kwargs["Ru"]/kwargs["M_ox"]
    T_ox = kwargs["T_ox"]
    Df = kwargs["Df"]
    a = kwargs["a"]
    rho_ox = Pc/(R_ox*T_ox)
    Vox = mox/(rho_ox*(1-a)*np.pi*np.power(Df,2)/4)
    return(Vox)


def func_R(of, Pc, func_M):
    """ Interpolate fuction of gas constant
    
    Parameter
    ------------
    of: float
        O/F
    Pc: float
        chamber pressure [Pa]
    
    Return
    -------------
    R: float
        gas constant [J/kg/K]
    """
    Rstr = 8.3144598 # gas constant [J/mol/K]
    M = func_M(of,Pc)*1.0e-3
    R = Rstr/M
    # R = func_cp(of, Pc)*1.0e+3*(1-1/func_gamma(of, Pc))
    return(R)


def func_dPc(t, Pc, mox, val, t_history, Vf_history, **cond):
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

    Vox = func_Vox(Pc, mox, **cond)
    Vf = func_Vf(Pc, Vox)
    val["Pc"] = Pc
    val["Vox"] = Vox
    r_tmp, rdot_tmp, rdotn_tmp = mod_shape.exe(val, **cond)
    r, rdot, rdotn = mod_shape.func_rcut(r_tmp, rdot_tmp, rdotn_tmp, t_history, Vf_history, **cond)
    if cond["Vf_mode"]:
        mf = N *mod_shape.func_mf(r[~np.isnan(r)].size-1, r[~np.isnan(r)], rdot[~np.isnan(rdot)], Vf=Vf ,**cond)    
    else:
        mf = N *mod_shape.func_mf(r[~np.isnan(r)].size-1, r[~np.isnan(r)], rdot[~np.isnan(rdot)], **cond)
    of = mox/mf
    cstr = func_cstr(of, Pc)
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


def exe_RK4(t, val, func_mox, t_history, Vf_history, **cond):
    dt = cond["dt"]
    Pc = val["Pc"]
    cond_RK4 = copy.deepcopy(cond)   # copy cond to exchange dt for each k calculation
    cond_RK4["dt"] = 0.0        # set appropriate dt for shape calculation
    k0 = dt*func_dPc(t, Pc, func_mox(t), val, t_history, Vf_history, **cond_RK4)
    cond_RK4["dt"] = dt/2       # set appropriate dt for shape calculation
    k1 = dt*func_dPc(t+dt/2, Pc+k0/2, func_mox(t+dt/2), val, t_history, Vf_history, **cond_RK4)
    cond_RK4["dt"] = dt/2       # set appropriate dt for shape calculation
    k2 = dt*func_dPc(t+dt/2, Pc+k1/2, func_mox(t+dt/2), val, t_history, Vf_history, **cond_RK4)
    cond_RK4["dt"] = dt         # set appropriate dt for shape calculation
    k3 = dt*func_dPc(t+dt, Pc+k2, func_mox(t+dt), val, t_history, Vf_history, **cond_RK4)
    k = (k0 +2*k1 +2*k2 +k3)/6
    Pc_new = Pc +k
    return Pc_new

