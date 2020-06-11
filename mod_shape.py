# coding: utf-8

"""
Content: Fuel Regression Shape of EBHR at transinet process
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

def func_Vf(Vox, Pc):
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
    C1 = 1.34e-7
    C2 = 1.61e-9
    n=0.951
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
    if "Vf" not in kwargs:
        mf = 2*np.pi*rho_f*simps((r_range + d/2)*rdot_range, x_range) # calculate using integretion for axial direction
    else:
        Vf = kwargs["Vf"]
        mf = rho_f*Vf*np.pi*(np.power(r[i], 2) +d*r[i] ) # calculate using integretion for radial direction
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
    Rox = kwargs["Ru"]/kwargs["M_ox"]
    Tox = kwargs["T_ox"]
    rho_ox = Pc/(Rox*Tox)
    Df = kwargs["Df"]
    Af = np.pi*np.power(Df, 2)/4
    d = kwargs["d"]
    N = kwargs["N"]
    a = 1 -np.power(d/Df, 2)*N
    Vox = mox/(rho_ox*Af*(1-a))
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
    if kwargs["Vf_mode"]:
        Vf = func_Vf(Vox, Pc)
        G = (func_mf(i, r, rdot, Vf=Vf, **kwargs) + func_mox(Vox, Pc, **kwargs)) / (np.pi*np.power(2*ri+d, 2))  # calculate using integretion for radial direction
    else:
        G = (func_mf(i, r, rdot, **kwargs) + func_mox(Vox, Pc, **kwargs)) / (np.pi*np.power(2*ri+d, 2))  # calculate using integretion for axial direction
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
        # rdot0 = Cr*np.power(G, z)*np.power(x[i], m)
        rdot0 = 10.0e-6*np.power(G, z)*np.power(self.x[i], m)*np.power(self.Pc*1.0e-5, -1)
    th = k*np.power(G, 0.8)/Pc
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
    Vf = func_Vf(Vox, Pc)
    if x[i] == 0:
        rdot_norm = 0.0
    else:
        theta = np.arctan((ri-r[i-1])/(2*dx))
        rdot_norm = Vf*np.tan(theta)
    return(rdot_norm)

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
    f = -func_Vf(Vox, Pc)*drdx + func_rdot(i, ri, r, val, **kwargs)
    return(f)

def RK4(i, val, **kwargs):
    """ 4th Runge-Kutta method.

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
    pitch = kwargs["pitch"]
    d = kwargs["d"]
    r = val["r"]
    dt = kwargs["dt"]
    # flag = False
    # for j in range(i): 
    #     if np.isnan(r[j]):
    #         flag = True
    #         break
    #     else:
    #         flag = False
    # if flag:    # if np.nan is included befoer No.i at No.j, later value from No.j of r is np.nan
    #     r_new = np.nan
    # else:
    k1 = dt* func_f(r[i], i, val, **kwargs)
    k2 = dt* func_f((r[i]+k1)/2, i, val, **kwargs)
    k3 = dt* func_f((r[i]+k2)/2, i, val, **kwargs)
    k4 = dt* func_f(k3, i, val, **kwargs)
    k = (k1 + 2*k2 + 2*k3 + k4)/6
    r_new = r[i] + k
        # if r_new + d/2 > pitch/2: # if regressed fuel port diamter exceeds half of pitch, insert nan value
        #     r_new = np.nan
    return(r_new)

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
        r_new[i] = RK4(i, val, **kwargs) # calculate fuel regressin distance from inner port surface
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

def func_gen_imgfile(r, rdot, img_list, **kwargs):
    """ Stuckking the image file to img_list
    
    Parameters
    ----------
    r : 1d-ndarray of float
        radial fuel regression distance
    rdot : 1d-ndarray of float
        radial fuel regression rate
    img_list : list of matplotlib.pyplot.plot
        list of plot image which contains radial fuel regression plot and one more other plot
    
    Return
    ----------
    img_list: list of matplotlib.pyplot.plot
        image list after stacked new plot image
    """
    x_max = kwargs["x_max"]
    x = kwargs["x"]
    t_end = kwargs["t_end"]
    pitch = kwargs["pitch"]
    title = ax1.text(x_max/2*1e+3, y_max*1.1*1e+3, "t={} s".format(round(t,3)), fontsize="large")
    ax1.set_xlabel("Axial distance $x$ [mm]")
    ax1.set_ylabel("Radial regression distance $r$ [mm]")
    ax1.set_ylim(-y_max*1.0e+3, y_max*1.0e+3)
    ax1.set_xlim(-x_max*1.0e+3,x_max*1.0e+3)
    ax1.grid()
    img1 = ax1.plot(np.append(x_front, x)*1.0e+3, (np.append(R_front, r)+d/2)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, -(np.append(R_front, r)+d/2)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, (np.append(R_front, r)+d/2 +pitch)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, -(np.append(R_front, r)+d/2 +pitch)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, (np.append(R_front, r)+d/2 -pitch)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, -(np.append(R_front, r)+d/2 -pitch)*1.0e+3, color="b")
#     ax2.set_xlabel("Axial distance $x$ [mm]")
    ax2.set_xlabel("Time $t$ [s]")
#     ax2.set_ylabel("Radial regression rate $\dot r$ [mm/s]")
    ax2.set_ylabel("Fuel mass flow rate $\dot m_f$ [g/s]")
#     ax2.set_ylim(0, 2.0)
    ax2.set_ylim(0, 10)
#     ax2.set_xlim(-x_max*1.0e+3,x_max*1.0e+3)
    ax2.set_xlim(0, t_end)
    ax2.grid()
#     img2 = ax2.plot(np.append(x_front, x)*1.0e+3, np.append(Rdot_front, rdot)*1.0e+3, color="r") # Radial fuel regression rate v.s. x
    img2 = ax2.plot(t_history, MF_history*1.0e+3, color="r") # Fuel mass flow rate v.s. t
    img_list.append(img1 + img2 + [title])
    return img_list

def plot_result(r, rdot, **kwargs):
    """ output image files when calculation was finished
    
    Parameters
    ----------
    r : 1d-array of float
        radial fuel regression distance
    rdot : 1d-ndarray of float
        radial fuel regression rate
    """
    x_max = kwargs["x_max"]
    x = kwargs["x"]
    t_end = kwargs["t_end"]
    pitch = kwargs["pitch"]
    fig1 = plt.figure()
    fig1ax= fig1.add_subplot(111)
    fig1ax.plot(np.append(x_front, x)*1.0e+3, (np.append(R_front, r)+d/2)*1.0e+3, color="b")
    fig1ax.plot(np.append(x_front, x)*1.0e+3, -(np.append(R_front, r)+d/2)*1.0e+3, color="b")
    fig1ax.plot(np.append(x_front, x)*1.0e+3, (np.append(R_front, r)+d/2+pitch)*1.0e+3, color="b")
    fig1ax.plot(np.append(x_front, x)*1.0e+3, (pitch-(np.append(R_front, r)+d/2))*1.0e+3, color="b")
    fig1ax.plot(np.append(x_front, x)*1.0e+3, (np.append(R_front, r)+d/2-pitch)*1.0e+3, color="b")
    fig1ax.plot(np.append(x_front, x)*1.0e+3, -(np.append(R_front, r)+d/2+pitch)*1.0e+3, color="b")
    fig1ax.set_xlabel("Axial distance $x$ [mm]")
    fig1ax.set_ylabel("Radial regression distance $r$ [mm]")
    fig1ax.set_ylim(-y_max*1.0e+3, y_max*1.0e+3)
    fig1ax.set_xlim(-x_max*1.0e+3, x_max*1.0e+3)
    fig1ax.set_title("t={} s".format(round(t,4)))
    fig1ax.grid()
    fig1.savefig(os.path.join(fld_name,"r_end.png"))
    fig1.show()
    fig2 = plt.figure()
    fig2ax= fig2.add_subplot(111)
    # fig2ax.plot(np.append(x_front, x)*1.0e+3, np.append(Rdot_front, rdot)*1.0e+3, color="r")
    fig2ax.plot(t_history, MF_history*1.0e+3, color="r")
    # fig2ax.set_xlabel("Axial distance $x$ [mm]")
    fig2ax.set_xlabel("Time $t$ [s]")
    # fig2ax.set_ylabel("Radial regression rate $\dot r$ [mm/s]")
    fig2ax.set_ylabel("Fuel mass flow rate $\dot m_f$ [g/s]")
    # fig2ax.set_ylim(0,)
    fig2ax.set_ylim(0, 10)
    # fig2ax.set_xlim(-x_max*1.0e+3,x_max*1.0e+3)
    fig2ax.set_xlim(0, t_end)
    fig2ax.set_title("t={} s".format(round(t,4)))
    fig2ax.grid()
    fig2.savefig(os.path.join(fld_name,"rdot_end.png"))

# %%
if __name__ == "__main__":
    cond={
        "d":  0.3e-3, # [m] port diameter
        "N": 433, # [-] the number of port
        "d_exit": 1.0e-3, # [m] port exit diamter
        "depth": 2.0e-3, # [m] depth of expansion region of port exit
        "pitch": 2.0e-3, # [m] pitch between each ports
        "rho_f": 1190, # [kg/m^3] solid fuel density
        "M_ox": 32.0e-3, # [kg/mol]
        "Rstr": 8.3144598, # [J/mol-K]
        "T": 300, # [K] oxidizer tempreature
        "Cr": 4.58e-6, # experimental value of radial fuel regression rate
        "z": 0.9,  # experimental value of radial fuel regression rate
        "m": -0.2,  # experimental value of radial fuel regression rate
        "k": 3.0e+4,  # experimental value of radial fuel regression rate
        "dt": 0.001, # [s] time step
        "dx": 0.1e-3, # [m] space resolution
        "t_end": 5.0, # [s] calculation end time
        "Vf_max": 10.0e-3, # [m] maximum fuel regression rate
        "x_max": 5.0e-3 # [m] maximum calculation region
    }
    
    CFL = np.abs(cond["Vf_max"]*cond["dt"]/cond["dx"])
    if CFL>=1.0:
        flag = False
    else:
        flag = True
    print("CFL condition = {}; CFL = {}".format(flag, CFL))


    def func_Pc(t):
        Pc = 1.0e+6 # [MPa] chamber pressure
        return(Pc)

    # def func_Vox(t):
    #     Vox = 30 # [m/s] oxidizer port velocity
    #     if t<2.0:
    #         val = Vox
    #     else:
    #         val = 2*Vox
    #     return val

    def func_mox(t):
        mox1 = 3.0 # [g/s] oxidizer mass flow rate before slottling
        mox2 = 6.0 # [g/s] oxidizer mass flow rate after slottling
        if t < 2.0:
            mox = mox1
        else:
            mox = mox2
        return mox

    fld_name = datetime.now().strftime("%Y_%m%d_%H%M%S")
    os.makedirs(fld_name)
    y_max = 5.0e-3
    plot_interval = 0.01 # [s] plot interval
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(wspace=0.3)
    img_list = []
    Vf_history = np.array([])
    MF_history = np.array([])
    MOX_history = np.array([])

    cond["t"], cond["x"], R_tmp, Rdot_tmp, Rdotn_tmp, R_front = initialize_calvalue(**cond)
    dt = cond["dt"]
    N = cond["N"]
    d = cond["d"]
    x_front = np.sort(-cond["x"])   # position value of uppstream from x=0
    val = {"r": R_tmp,
           "rdot": Rdot_tmp,
           "rdotn": Rdotn_tmp}

    for t in tqdm(cond["t"]):
        mox = func_mox(t)
        Pc = func_Pc(t)
        Vox = func_Vox(mox, Pc, **cond)
        val["Pc"] = Pc
        val["Vox"] = Vox
        R_tmp, Rdot_tmp, Rdotn_tmp = exe(val, **cond)
        Vf = func_Vf(Vox, Pc)
        Vf_history = np.append(Vf_history, Vf)
        t_history = np.arange(0, t+dt/2, dt)
        R, Rdot, Rdotn = func_rcut(R_tmp, Rdot_tmp, Rdotn_tmp, t_history, Vf_history, **cond)
        MF =  N *func_mf(R[~np.isnan(R)].size, R[~np.isnan(R)], Rdot[~np.isnan(Rdot)], **cond)
        MF_history = np.append(MF_history, MF)
        MOX = N *np.pi*np.power(d, 2)/4 *func_Vox(t)
        MOX_history = np.append(MOX_history, MOX)
        if int(t/dt) % int(plot_interval/dt) == 0 or t==0.0:
            img_list = func_gen_imgfile(R, Rdot, img_list, **cond)
        val["r"] = R_tmp
        val["rdot"] = Rdot_tmp
        val["rdotn"] = Rdotn_tmp
    
    # plot the last simulation result of fuel regression shape and radial regression rate.
    plot_result(R, Rdot, **cond)
    # generate animation
    anim = ArtistAnimation(fig, img_list, interval=dt*1e+3)
    anim.save(os.path.join(fld_name, "animation.mp4"), writer="ffmpeg", fps=10)


