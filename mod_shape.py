# coding: utf-8

"""
Content: Module for Fuel Regression Shape of EBHR at transinet process
Author: Ayumu Tsuji @Hokkaido University

Description:
Estimate the single port fuel regression shape of axial-injection end-burning
hybrid rocket at transient process
To estimate the shape, this program needs some empirical constant,Cr; z; m, k,
which are obtained by experimental results.
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import ArtistAnimation
from scipy.integrate import simps
from scipy.optimize import newton
from tqdm import tqdm
from datetime import datetime
import os
import copy

# %%
def initialize_calvalue(**kwargs):
    """ Generate and initialize variable vector; x, r, rdot, rdotn

    Parameter
    ---------
    **kwargs: dictionary
        calculation parameter set

    Return
    --------
    t: id-ndarray of float
        time [s]
    x: 1d-ndarray of float
        position [m]
    r: 1d-ndarray of float
        radial regression distance [m]
    rdot: 1d-ndarray of float
        radial regression rate [m/s]
    rdotn: 1d-ndarray of float
        regression rate which is nromal to regression surface [m/s]
    r_front: 1d-ndarray of float [m]
        radial regression distance, which is normally zero, upstream on regression tip [m]
    """
    t_end = kwargs["t_end"]
    dt = kwargs["dt"]
    x_max = kwargs["x_max"]
    dx = kwargs["dx"]
    d_exit = kwargs["d_exit"]
    d = kwargs["d"]
    depth = kwargs["depth"]
    # Generate variable
    t = np.arange(0, t_end+dt, dt)
    x = np.arange(0.0, x_max+dx, dx)
    r = np.zeros(int(round((x_max+dx)/dx,0)), float)
    rdot = np.zeros(int(round((x_max+dx)/dx)), float)
    rdotn = np.zeros(int(round((x_max+dx)/dx)), float)
    # Boundary condition
    r_0 = 0 # r=0 when x = 0
    rdot_0 = 0.0 # rdot=0 when x = 0
    rdotn_0 = 0.0 # rdotn=0 when x = 0
    # Initialize
    for i in range(x.size):
        r[i] = ((d_exit-d)/2)/depth * x[i]
    rdot = np.array([rdot_0 for i in rdot])
    rdotn = np.array([rdotn_0 for i in rdotn])
    return t, x, r, rdot, rdotn

def func_Vf(Vox, Pc, **kwargs):
    """ Axial fuel regression rate

    Parameter
    ---------
    Vox: float
        oxidizer port velocity [m/s]
    Pc: float
        chamber pressure [Pa]

    Return
    ---------
    Vf :float
        axial fuel regression rate [m/s]
    """
    C1 = kwargs["C1"]
    C2 = kwargs["C2"]
    n = kwargs["n"]
    Vf = (C1/Vox + C2)*np.power(Pc, n)
    return(Vf)

def func_mf(i, r, rdot, **kwargs):
    """ cumlative fuel mass flow rate evaporated from adress 0 to i.

    Parameter
    ----------
    i: int
        address number
    r: 1d-ndarray
        array of radial regression distance [m]
    rdot: 1d-ndarray
        array of radial regression rate [m]
    **kwargs: dictionary
        calculation parameter set

    Return
    ---------
    mf: float
        cumulative fuel mass flow rate evaporated from address 0 to i [kg/s]
    """
    rho_f = kwargs["rho_f"]
    d = kwargs["d"]
    x = kwargs["x"]
    if i == 0:
        x_range = np.array([0.0])
        r_range = np.array([0.0])
        rdot_range = np.array([0.0])
    else:
        x_range = x[:i]
        r_range = r[:i]
        rdot_range = rdot[:i]
    if kwargs["Vf_mode"]:
        Vf = kwargs["Vf"]
        mf = rho_f*Vf*np.pi*(np.power(r[i], 2) +d*r[i] ) # calculate using integretion for radial direction
    else:
        Vf = kwargs["Vf"]
        mf = 2*np.pi*rho_f*simps((r_range + d/2)*rdot_range, x_range) # calculate using integretion for axial direction
    return(mf)

def func_mox(Vox, Pc, **kwargs):
    """ oxidizer mass flow rate at each port

    Parameter
    ------------
    Vox: float
        oxidizer port velocity [m/s]
    Pc: float
        chamber pressure [Pa]
    **kwargs: dictionary
        calculation parameter set
    
    Return
    ------------
    mox: float
        oxidizer mass flow rate at each port
    """
    M_ox = kwargs["M_ox"]
    T_ox = kwargs["T_ox"]
    Ru = kwargs["Ru"]
    d = kwargs["d"]
    rho_ox = Pc/((Ru/M_ox)*T_ox)
    mox = rho_ox*Vox*(np.pi*np.power(d,2)/4)
    return(mox)

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

def func_G(i, ri, r, rdot, val, **kwargs):
    """ Local propellant mass flux

    Parameter
    ----------
    i: int
        location address for axial down stream direction
    ri: float
        local regression distance from inner port surface [m]
    r: 1d-ndarray
        regression distance [m]
    rdot: 1d-ndarray
        regression rate [m]
    **kwargs: dictionary
        calculation parameter set
    
    Return
    ----------
    G: float
        Local propellant mass flux [kg/m^2/s]
    """
    d = kwargs["d"]
    Vox = val["Vox"]
    Pc = val["Pc"]
    Vf = func_Vf(Vox, Pc, **kwargs)
    G = (func_mf(i, r, rdot, Vf=Vf, **kwargs) + func_mox(Vox, Pc, **kwargs)) / (np.pi*np.power(2*ri+d, 2)/4)  # calculate using integretion for radial direction
    return(G)

def func_rdot(i, ri, r, val, **kwargs):
    """ Local radial regression rate

    Parameter
    ----------
    i: int
        location address for axial down stream direction
    ri: float
        local regression distance from inner port surface [m]
    r: 1d-ndarray
        regression distance [m]
    val : dictionary
        dictionary set of variables
        {"r": 1d-ndarray; radial regression,
        "rdot": 1d-ndarray; radial regression rate,
        "rdotn": 1d-ndarray; normal regressionrate,
        "Pc": float; chamber pressure,
        "Vox": float; oxidizer port velocity}
    **kwargs: dictionary
        calculation parameter set
    
    Return
    ----------
    rdot: float
        Local radial regression rate [m/s]
    """
    x = kwargs["x"]
    Pc = val["Pc"]
    Vox = val["Vox"]
    k = kwargs["Cr"]
    z = kwargs["z"]
    m = kwargs["m"]
    k = kwargs["k"]
    Cr = kwargs["Cr"]
    rdot = val["rdot"]
    G = func_G(i, ri, r, rdot, val, **kwargs)
    if x[i] == 0:
        rdot0 = 0.0
    else:
        rdot0 = Cr*np.power(G, z)*np.power(x[i], m)
    # th = k*np.power(G, 0.8)/Pc
    # rdoti = rdot0 *np.sqrt(2/th) *np.sqrt(1 -1/th*(1 -np.exp(-th)))
    rdoti = rdot0
    return(rdoti)

def func_rdotn(i, ri, r, val, **kwargs):
    """ Local normal regression rate

    Parameter
    ----------
    i: int
        location address for axial down stream direction
    ri: float
        local regression distance from inner port surface [m]
    r: 1d-array
        radial regression distance [m]
    val : dictionary
        dictionary set of variables
        {"r": 1d-ndarray; radial regression,
        "rdot": 1d-ndarray; radial regression rate,
        "rdotn": 1d-ndarray; normal regressionrate,
        "Pc": float; chamber pressure,
        "Vox": float; oxidizer port velocity}
    **kwargs: dictionary
        calculation parameter set
    
    Return
    ----------
    rdot_norm: float
        Local normal regression rate [m/s]
    """
    x = kwargs["x"]
    dx = kwargs["dx"]
    Pc = val["Pc"]
    Vox = val["Vox"]
    Vf = func_Vf(Vox, Pc, **kwargs)
    if x[i] == 0:
        rdot_norm = 0.0
    else:
        theta = np.arctan((ri-r[i-1])/(2*dx))
        rdot_norm = Vf*np.tan(theta)
    return(rdot_norm)

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

def func_f(ri, i, val, **kwargs):
    """ delivative of Runge-Kutta function

    Parameter
    ----------
    ri: float
        local regression distance from inner port surface [m]
    i: int
        address number
    val : dictionary
        dictionary set of variables
        {"r": 1d-ndarray; radial regression,
        "rdot": 1d-ndarray; radial regression rate,
        "rdotn": 1d-ndarray; normal regressionrate,
        "Pc": float; chamber pressure,
        "Vox": float; oxidizer port velocity}
    **kwargs: dictionary
        calculation parameter set

    Return
    ------------
    f: float
        deligative of Runge-Kutta function
    """
    dx = kwargs["dx"]
    Vox = val["Vox"]
    Pc = val["Pc"]
    r = val["r"]
    if i == 0:
        drdx = (r[i]-0.0)/dx
    else:
        drdx = (r[i]-r[i-1])/dx
    f = -func_Vf(Vox, Pc, **kwargs)*drdx + func_rdot(i, ri, r, val, **kwargs)
    return(f)

def EULER(i, val, **kwargs):
    """ Euler explicit method.

    Parameter
    ----------
    i: int
        address number
    val : dictionary
        dictionary set of variables
        {"r": 1d-ndarray; radial regression,
        "rdot": 1d-ndarray; radial regression rate,
        "rdotn": 1d-ndarray; normal regressionrate,
        "Pc": float; chamber pressure,
        "Vox": float; oxidizer port velocity}
    **kwargs: dictionary
        calculation parameter set
    
    Return
    ----------
    r_new: float
        new radial regression distance at No.i element
    """
    r = val["r"]
    dt = kwargs["dt"]
    r_new = r[i] + dt*func_f(r[i], i, val, **kwargs)
    return r_new

def exe(val, **kwargs):
    """Execute single time step calculation of regression shape
    
    Parameters
    ----------
    val : dictionary
        dictionary set of variables
        {"r": 1d-ndarray; radial regression,
        "rdot": 1d-ndarray; radial regression rate,
        "rdotn": 1d-ndarray; normal regressionrate,
        "Pc": float; chamber pressure,
        "Vox": float; oxidizer port velocity}
    **kwargs: dictionary
        dictionary of calculation condition

    Return
    -------
    val: dictionary
        updated dictionary set of variables
    """
    x_max = kwargs["x_max"]
    dx = kwargs["dx"]
    r = val["r"]
    r_new = np.zeros(int(round((x_max+dx)/dx)), float)
    rdot_new = np.zeros(int(round((x_max+dx)/dx)), float)
    rdotn_new = np.zeros(int(round((x_max+dx)/dx)), float)
    for i in range(len(r)):
        r_new[i] = EULER(i, val, **kwargs) # calculate fuel regressin distance from inner port surface
        rdot_new[i] = func_rdot(i, r_new[i], r_new, val, **kwargs)
        rdotn_new[i] = func_rdotn(i, r_new[i], r_new, val, **kwargs)
    return r_new, rdot_new, rdotn_new

def func_rcut(r_tmp, rdot_tmp, rdot_norm_tmp, t_history, Vf_history, **kwargs):
    """ insert Nan value when the neiboring ports merge and port exit reaches end-surface
    
    Parameters
    ----------
    r_tmp : 1d-ndarray
        radial regression distance before cutting
    rdot_tmp : 1d-ndarray
        radial regression rate before cutting
    rdot_norm_tmp : [type]
        normal regression distance before cutting
    t_history : 1d-ndarray
        time history
    Vf_history : 1d-ndarray
        axial fuel regression history
    
    Returns
    -------
    r: 1d-ndarray
        fuel regression distance after cutting
    rdot: 1d-ndarray
        radial fuel regression rate after cutting
    rdotn_norm: 1d-ndarray
        normal fuel regression distance after cutting    
    """
    Lx = simps(Vf_history, t_history) # regression distance for axial direction
    r = copy.deepcopy(r_tmp)
    rdot = copy.deepcopy(rdot_tmp)
    rdot_norm = copy.deepcopy(rdot_norm_tmp)
    x = kwargs["x"]
    d = kwargs["d"]
    depth = kwargs["depth"]
    pitch = kwargs["pitch"]
    for i in range(x.size):
        if r[i] + d/2 > pitch/2: # if regressed fuel port diamter exceeds half of pitch, insert nan value
            r[i] = np.nan
            rdot[i] = np.nan
            rdot_norm[i] = np.nan
        elif x[i] > (Lx+depth): # when r exceeds end of fuel sureface, insert nan value 
            r[i] = np.nan
            rdot[i] = np.nan
            rdot_norm[i] = np.nan            
        else:
            r[i] = r_tmp[i]
            rdot[i] = rdot_tmp[i]
            rdot_norm[i] = rdot_norm_tmp[i]
    return r, rdot, rdot_norm