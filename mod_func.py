# coding: utf-8

"""
Content: Module for the funcsion list of combustion parameters of EBHR
Author: Ayumu Tsuji @Hokkaido University

Description:
At the following part, several function are listed to calculate combustion paremters
such as Re number, firction velocity, thrust and so on.
"""

import numpy as np

def func_Vox(mox, Pc, **kwargs):
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
    return Vox

def func_re(P, u, **kwargs):
    T = kwargs["T_ox"]
    Rm = kwargs["Ru"]/kwargs["M_ox"]
    d = kwargs["d"]
    mu = kwargs["mu_ox"]
    rho = P/(Rm*T)
    Re = rho*u*d/mu
    return Re

def func_ustr_lam(P, u, **kwargs):
    T = kwargs["T_ox"]
    Rm = kwargs["Ru"]/kwargs["M_ox"]
    d = kwargs["d"]
    mu = kwargs["mu_ox"]
    rho = P/(Rm*T)
    grad = 4*u/d
    tau = mu*grad
    ustr = np.sqrt(tau/rho)
    return ustr

def func_ustr_turb(P, u, **kwargs):
    T = kwargs["T_ox"]
    Rm = kwargs["Ru"]/kwargs["M_ox"]
    d = kwargs["d"]
    mu = kwargs["mu_ox"]
    rho = P/(Rm*T)
    nu = mu/rho
    lmbd = 0.3164*np.power(u*d/nu, -1/4)
    tau = lmbd*rho*np.power(u, 2)/8
    ustr = np.sqrt(tau/rho)
    return ustr
