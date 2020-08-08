# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:56:46 2018

@author: TJ
"""

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm
from cea_post import Read_datset

#%%
"""##########################
# Define unchange parameter #  
##########################"""
Df = 38e-3 #　[m] fuel outer diamter
a = 0.98 #　[-] fuel filling rate
Dt = 6.2e-3 # [m] nozzle thorat diamter
rho_f = 1191 # [kg/m^3] fuel density
R_ox = 8.43144598/32e-3 #　[J/kg/K] oxidizer gas constant
T_ox = 300 #　[K] oxidizer temperature

# %%
"""#####################
# Define initial value #  
#####################"""
Lci = 20.0e-3 #　[m] initial chamber length
Vci = np.pi*np.power(Df,2)/4 * Lci #　[m^3] initial chamber volume

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
    n = 0.951
    C1  = 1.39e-7
    C2 = 1.61e-9
    Vf = (C1/Vox + C2)*np.power(Pc,n)
    return(Vf)

def func_Vox( Pc, mox):
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
    rho_ox = Pc/(R_ox*T_ox)
    Vox = mox/(rho_ox*(1-a)*np.pi*np.power(Df,2)/4)
    return(Vox)

cea_fldpath = os.path.join("cea_db", "GOX_CurableResin", "csv_database") # assign folder path of cea-database
func_T = Read_datset(cea_fldpath).gen_func("T_c") # generate gas temeratur interporate function
func_CSTAR = Read_datset(cea_fldpath).gen_func("CSTAR") # generate c* interporate functioni
func_M = Read_datset(cea_fldpath).gen_func("M_c") # generate molecular weight interpolate function

def func_R(of, Pc):
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
    Rstr = 8.43144598 # gas constant [J/mol/K]
    M = func_M(of,Pc)
    R = Rstr/M
    return(R)


def func_diff(t, Pc, t_array, Vf_array, func_mox):
    """ Delivative of Pc 
    
    Parameter
    ------------
    t: float
        time [s]
    Pc: float
        chamber pressure [Pa]
    func_mox: function, func(t).
        oxidizer mass flow rate, function of t.
    
    Return
    ----------
    dPc: float
        delivative of chamber pressure [Pa/s]
    """
    Af = np.pi*np.power(Df,2)/4
    At = np.pi*np.power(Dt,2)/4
    mox = func_mox(t)
    Vox = func_Vox(Pc, mox)
    Vf = func_Vf(Pc, Vox)
    mf = Vf*a*Af*rho_f
    of = mox/mf
    cstr = func_CSTAR(of,Pc)
    T = func_T(of,Pc)
    R = func_R(of,Pc)
    if t_array[-1] == t:
        t_array_tmp = t_array
        Vf_array_tmp = Vf_array
    else:
        t_array_tmp = np.append(t_array, t)
        Vf_array_tmp = np.append(Vf_array, Vf)
    integ_Vf = integrate.simps(Vf_array_tmp, t_array_tmp)
    
    dPc = (-1*Af*Pc*Vf/(R*T) +a*rho_f*Af*Vf -Pc*At/cstr +mox)/((Vci+Af*integ_Vf)/(R*T))
    return(dPc)

def exe_rungekutta(t_range, func_mox):
    # initialize each parameter
    Pc_ = Pci
    t_array = np.array([])
    Pc_array = np.array([])
    mox_array = np.array([])
    Vox_array = np.array([])
    Vf_array = np.array([])
    mf_array = np.array([])
    of_array = np.array([])
    cstr_array = np.array([])
    T_array = np.array([])
    R_array = np.array([])
    integ_Vf_array = np.array([])
    for t in tqdm(t_range):
        Pc = Pc_
        Af = np.pi*np.power(Df,2)/4
        mox = func_mox(t)
        Vox = func_Vox(Pc, mox)
        Vf = func_Vf(Pc, Vox)
        mf = Vf*a*Af*rho_f
        of = mox/mf
        cstr = func_CSTAR(of,Pc)
        T = func_T(of,Pc)
        R = func_R(of,Pc)
        t_array = np.append(t_array, t)
        Pc_array = np.append(Pc_array, Pc)
        mox_array = np.append(mox_array, mox)
        Vox_array = np.append(Vox_array, Vox)
        Vf_array = np.append(Vf_array, Vf)
        mf_array = np.append(mf_array, mf)
        of_array = np.append(of_array, of)
        cstr_array = np.append(cstr_array, cstr)
        T_array = np.append(T_array, T)
        R_array = np.append(R_array, R)
        integ_Vf = integrate.simps(Vf_array, t_array)
        integ_Vf_array = np.append(integ_Vf_array, integ_Vf)
        
        k0 = dt*func_diff(t, Pc, t_array, Vf_array, func_mox)
        k1 = dt*func_diff(t+dt/2, Pc+k0/2, t_array, Vf_array, func_mox)
        k2 = dt*func_diff(t+dt/2, Pc+k1/2, t_array, Vf_array, func_mox)
        k3 = dt*func_diff(t+dt, Pc+k2, t_array, Vf_array, func_mox)
        k = (k0 +2*k1 +2*k2 +k3)/6
        Pc_ = Pc +k     
    
    df = pd.DataFrame([], index=t_array)
    df["mox"] = mox_array
    df["Pc"] = Pc_array
    df["Vox"] = Vox_array
    df["Vf"] = Vf_array
    df["mf"] = mf_array
    df["of"] = of_array
    df["cstr"] = cstr_array
    df["Tg"] = T_array
    df["Rg"] = R_array
    df["integ_Vf"] = integ_Vf_array
    return(df)

# %%
"""#################
# Calculation Part #  
#################"""

# %%
"""#######
# Part 1 #
#######"""
# define the function of oxidizer mass flow rate; func_mox
def func_mox(t):
    init = 15.0 # [s] transient supresstion time
    period = 20.0 # [s] period
    d_ratio = 0.5 # [-] duty ratio
    mox_max = 11.0e-3 # [kg/s] maximum oxidizer mass flow rate
    mox_min = 10.0e-3 # [kg/s] minimum oxidizer mass flow rate
    if t<init:
        mox = mox_min
    else:
        if (t+init)%period < period*d_ratio:
            mox = mox_max
        else:
            mox = mox_min
    return(mox)

# define the calculation time step
dt = 0.005 # [s] time step

# define the calculation time
t_range = np.arange(0, 60 +dt/2, dt)

# define the initial chamber pressure
Pci = 0.1013e+6 #　[Pa] initial chamber pressure, absolute pressure

dat = exe_rungekutta(t_range,func_mox)