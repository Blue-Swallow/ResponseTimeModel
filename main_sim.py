# -*- coding: utf-8 -*-
"""
# Content: Firing Simulator of Axial-Injection End-Buring Hybrid Rocket
# Author: Ayumu Tsuji @Hokkaido University

# Description:
Simulate several parameters and fuel regression shape at firing test
of axial-injection end-buring hybrid rocket. The key technology is that
a fuel regression shape is calculated by using non-steady advection equation.
This method enable to reproduce a long response time at throttling operation.
"""
# %%
import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from cea_post import Read_datset
import mod_response
import mod_shape
import mod_plot

# %%
class Main:
    """ class for the firing test simulation fo Axial-Injection End-Buring Hybrid Rocket

    Attributes
    ----------
    ## dictionary of parameter for simulation
    cond_ex : dic
        dictionary of experimental condition
    cond_cal : dic
        dictionary of calculation condition
    const_model: dic
        dictionary of regression model constants
    funclist_cea : dic
        dictionar of cea data-base information
    plot_param : dic
        dictionary of the parameters of plot and animation

    ## Array of calculated parameter history
    t_history : 1D-ndarray
        array of time
    Pc_history : 1D-ndarray
        array of chamber pressure history
    r_history : 2D-ndarray
        array of regression shape history
    rdot_history 2D-ndarray
        array of radial regression rate history
    rdotn_history : 2D-ndarray
        array of normal regression rate history
    Vf_history : 1D-ndarray
        array of axial regression rate history
    Vox_history : 1D-ndarray
        array of oxidizer port velocity history
    mf_history : 1D-ndarray
        array of fuel mass flow rate history
    mox_history 1D-ndarray
        array of oxidizer mass flow rate history
    cstr_history : 1D-ndarray
        array of characteristic exhaust velocity history
    of_history 1D-ndarray
        array of O/F history
    x: 1D-ndarray
        array of axial distance

    ## Others
    fld_name: str
        folder name in which the results is contained
    img_list : list of Figure objects of matplotlib
        image list which is used for generating animation.
    """

    def __init__(self, cond_ex, cond_cal, const_model, cea_fldpath, funclist_cea, plot_param, fldname=False):
        """        
        Parameters
        ----------
        cond_ex : dic
            dictionary of experimental condition
        cond_cal : dic
            dictionary of calculation condition
        const_model : dic
            dictionary of regression model constants
        cea_fldpath : string
            folder path of cea data-base
        funclist_cea : dic
            dictionar of cea data-base information
        plot_param : dic
            dictionary of the parameters of plot and animation
        fldname : string, Optional
            the name of folder which will contain calculation result, by default None
        """
        self.fld_name = fldname
        self.cond_ex = cond_ex
        self.cond_ex["a"] = 1 -np.power(cond_ex["d"]/cond_ex["Df"], 2)*cond_ex["N"]     # [-] fuel filling rate
        self.cond_ex["Vci"] = np.pi*np.power(cond_ex["Df"], 2)/4 * cond_ex["Lci"]       # [m^3] initial chamber volume
        self.cond_ex["R_ox"] = cond_ex["Ru"]/cond_ex["M_ox"]      # [J/kg-K] oxidizer gas constant
        self.cond_cal = cond_cal
        self.const_model = const_model
        self.cea_fldpath = cea_fldpath
        self.funclist_cea = funclist_cea
        self.plot_param = plot_param
        self._initialize_()
        self._mk_json_()

    def _initialize_(self):
        """ Initialize some class variable
        """
        x_max = self.cond_cal["x_max"]
        dx = self.cond_cal["dx"]
        if self.fld_name:
            pass
        else:
            self.fld_name = datetime.now().strftime("%Y_%m%d_%H%M%S")   # folder name which contain animation and figure of calculation result
            os.makedirs(self.fld_name)
        self.img_list = []  # define the list in wihch images of plot figure are stacked
        self.t_history = np.array([])
        self.Pc_history = np.array([])
        self.r_history = np.empty([0, int(round((x_max+dx)/dx,0))])
        self.rdot_history = np.empty([0, int(round((x_max+dx)/dx,0))])
        self.rdotn_history = np.empty([0, int(round((x_max+dx)/dx,0))])
        self.Vf_history = np.array([])
        self.Vox_history = np.array([])
        self.Re_history = np.array([])
        self.ustr_lam_history = np.array([])
        self.ustr_turb_history = np.array([])
        self.mf_history = np.array([])
        self.mox_history = np.array([])
        self.cstr_history = np.array([])
        self.of_history = np.array([])

    def _mk_json_(self):
        """ Make the json file which contains calculation conditions
        """
        dic_json = {"PARAM_EXCOND": self.cond_ex,
                    "PARAM_CALCOND": self.cond_cal,
                    "PARAM_MODELCONST": self.const_model,
                    "CEA_FLDPATH": self.cea_fldpath
                    }
        with open(os.path.join(self.fld_name, "cond.json"), "w") as f:
            json.dump(dic_json, f, ensure_ascii=False, indent=4)

    def exe(self, func_mox):
        """Excecute a calculation and contains the results to each class attribute
        
        Parameters
        ----------
        func_mox : function(time)
            function of time, which returns oxidizer mass flow rate [kg/s]
        """
        ## combine several dict of parameters
        cond = dict(self.cond_ex, **self.cond_cal, **self.const_model, **self.funclist_cea, **self.plot_param)
        ## set several constant, function and variables before calculation
        N = self.cond_ex["N"]
        func_cstr = cond["func_CSTAR"]
        cond["time"], cond["x"], r_tmp, rdot_tmp, rdotn_tmp = mod_shape.initialize_calvalue(**cond)
        self.x = cond["x"]
        val = {}
        ## Following iteration part is the main sectioin of this simulation program.
        for t in tqdm(cond["time"]):
            ## update each value at the follwoing lines
            self.t_history = np.append(self.t_history, t)
            mox = func_mox(t)
            self.mox_history = np.append(self.mox_history, mox)
            if t == 0:
                Pc = cond["Pci"]
            else:
                Pc = Pc_new
            val["Pc"] = Pc
            self.Pc_history = np.append(self.Pc_history, Pc)
            Vox = mod_shape.func_Vox(mox, Pc, **cond)
            val["Vox"] = Vox
            self.Vox_history = np.append(self.Vox_history, Vox)
            Vf = mod_shape.func_Vf(Vox, Pc, **cond)
            self.Vf_history = np.append(self.Vf_history, Vf)
            Re = mod_shape.func_re(Pc, Vox, **cond)
            self.Re_history = np.append(self.Re_history, Re)
            ustr_lam = mod_shape.func_ustr_lam(Pc, Vox, **cond)
            self.ustr_lam_history = np.append(self.ustr_lam_history, ustr_lam)
            ustr_turb = mod_shape.func_ustr_turb(Pc, Vox, **cond)
            self.ustr_turb_history = np.append(self.ustr_turb_history, ustr_turb)
            if t != 0:
                r_tmp = r_new_tmp
                rdot_tmp = rdot_new_tmp
                rdotn_tmp = rdotn_new_tmp
            ## reshape and eliminate the unneccesary part of regression shape.
            r, rdot, rdotn = mod_shape.func_rcut(r_tmp, rdot_tmp, rdotn_tmp, self.t_history, self.Vf_history, **cond)
            self.r_history = np.vstack((self.r_history, r))
            self.rdot_history = np.vstack((self.rdot_history, rdot))
            self.rdotn_history = np.vstack((self.rdotn_history, rdotn))
            ## calculate the others parameter at the following lines
            if cond["Vf_mode"]:
                mf =  N *mod_shape.func_mf(r[~np.isnan(r)].size-1, r[~np.isnan(r)], rdot[~np.isnan(rdot)], Vf=Vf, **cond)
            else:
                mf =  N *mod_shape.func_mf(r[~np.isnan(r)].size-1, r[~np.isnan(r)], rdot[~np.isnan(rdot)], Vf=Vf, **cond)
            if cond["Af_modify"]:
                coef_Afmod = (np.power(self.cond_ex["Df"], 2) - N*np.power(self.cond_ex["d"], 2))/(N*(np.power(self.cond_ex["pitch"],2)-np.power(self.cond_ex["d"], 2)))
                mf = mf*coef_Afmod
            else:
                pass
            self.mf_history = np.append(self.mf_history, mf)
            if mf<=0.0:
                of = np.nan
                cstr_ex = Pc*np.pi*np.power(cond["Dt"], 2)/(4*mox)
            else:
                of = mox/mf
                cstr_ex = cond["eta"]*func_cstr(of, Pc)
            self.of_history = np.append(self.of_history, of)
            self.cstr_history = np.append(self.cstr_history, cstr_ex)
            ## calculate the next time step values at the following lines
            val["r"] = r_tmp
            val["rdot"] = rdot_tmp
            val["rdotn"] = rdotn_tmp
            Pc_new = mod_response.exe_EULER(t, mf, Pc, func_mox, self.t_history, self.Vf_history, **cond)
            r_new_tmp, rdot_new_tmp, rdotn_new_tmp = mod_shape.exe(val, **cond)
        ## CFL [-] Courant number, which must be less than unity       
        self.cond_cal["CFL"] = np.abs(self.Vf_history.max()*self.cond_cal["dt"]/self.cond_cal["dx"])


    def gen_img_list(self, fig, **kwargs):
        """ Stuckking the image file to img_list
        
        Parameters
        ----------
        fig : Figure object of matplotlib
        intrv: float, optional
            plot interval [s]. If this parameters is not assigned, the class attribute of plot_param["interval"] will be used.
                        
        Return
        ----------
        self.img_list: list of Figure objects of matplotlib
            image list stacked new plot image
        """
        cond = dict(self.cond_ex, **self.cond_cal, **self.plot_param)
        dic_dat = {"x": self.x,
                   "t_history": self.t_history,
                   "r_history": self.r_history,
                   "Pc_history": self.Pc_history,
                   "Vox_history": self.Vox_history,
                   "mf_history": self.mf_history,
                   "mox_history": self.mox_history,
                   "rdot_history": self.rdot_history,
                   "cstr_history": self.cstr_history,
                   "of_history": self.of_history,
                   "Vf_history": self.Vf_history
                   }
        self.img_list = mod_plot.gen_img_list(fig, self.img_list, dic_dat, **cond)
        return self.img_list


def read_cond():
    """ Assign the name of folder which contains calculation condition and will contain calculation result.
    
    Returns
    -------
    fldname: string
        the name of folder
    dic_cond: dict of dict
        dictionary of the dictioinary which contains calculating condition
        keylist: "PARAM_EXCOND", "PARAM_CALCOND", "PARAM_MODELCONST", "CEA_FLDPATH", "CEA_FLDPATH", "CEA_FLDPATH", "FUNCLIST_CEA", "PARAM_PLOT"
    func_mox: function
        function for oxidizer mass flow rate. this is the function of time.
    """
    name_mox_csv = "mox.csv"
    name_cond_py = "cond.py" 
    fldname = input("Please input the name of folder which contains calculating condition such as {} or {}\n>>".format(name_cond_py, name_mox_csv))
    if os.path.exists(fldname):
        if os.path.exists(os.path.join(fldname, name_cond_py)):
            sys.path.append(os.path.join(os.path.dirname(__file__), fldname))
            import cond
            dic_cond = {"PARAM_EXCOND": cond.PARAM_EXCOND,
                        "PARAM_CALCOND": cond.PARAM_CALCOND,
                        "PARAM_MODELCONST": cond.PARAM_MODELCONST,
                        "CEA_FLDPATH": cond.CEA_FLDPATH,
                        "FUNCLIST_CEA": cond.FUNCLIST_CEA,
                        "PARAM_PLOT": cond.PARAM_PLOT
                        }
        else:
            print("There is no python module of calculation condition, \"{}\"".format(name_cond_py))
            sys.exit()
        if dic_cond["PARAM_CALCOND"]["use_mox_csv"]:
            if os.path.exists(os.path.join(fldname, name_mox_csv)):
                df_mox = pd.read_csv(os.path.join(fldname, name_mox_csv), header=0, skiprows=[1,])
                df_mox.mox = df_mox.mox * 1.0e-3    # convert unit [g/s] to [kg/s]
                func_tmp = CubicSpline(df_mox.t, df_mox.mox, bc_type="clamped", extrapolate=True)
                def func_mox(t):
                    """Function of oxidizer mass flow rate [kg/s]
                    
                    Parameters
                    ----------
                    t : float
                        time [s]
                    
                    Returns
                    -------
                    mox: float
                        oxidizer mass flow rate [kg/s]
                    """
                    mox = float(func_tmp(t))
                    return mox
            else:
                print("There is no csv data of mox history, \"{}\"".format(name_mox_csv))
                sys.exit()
        else:
            func_mox = cond.FUNC_MOX    # read function of mox history from cond.py

    else:
        print("There is no such a folder, \"{}\"".format(fldname))
        sys.exit()
    return fldname, dic_cond, func_mox




# %%
if __name__ == "__main__":
    ##### Following Commented Section is to Conduct Debugging #####

    # # Parameters of experimental condition
    # PARAM_EXCOND = {"d": 0.272e-3,        # [m] port diameter
    #                 "N": 433,           # [-] the number of port
    #                 "Df": 38e-3,        # [m] fuel outer diameter
    #                 "d_exit": 1.0e-3,   # [m] port exit diameter
    #                 "depth": 2.0e-3,    # [m] depth of expansion region of port exit
    #                 "pitch": 1.7e-3,    # [m] pitch between each ports
    #                 "Dt": 6.2e-3,       # [m] nozzle throat diameter
    #                 # "Dt": 6.5e-3,       # [m] nozzle throat diameter
    #                 "Lci": 20.0e-3,     # [m] initial chamber length
    #                 "rho_f": 1190,      # [kg/m^3] solid fuel density
    #                 "M_ox": 32.0e-3,    # [kg/mol] oxidizer mass per unit mole
    #                 "T_ox": 300,        # [K] oxidizer temperature
    #                 "Ru": 8.3144598,    # [J/mol-K] Universal gas constant
    #                 "Pci": 0.1013e+6,   # [Pa] initial chamber pressure
    #                 "eta": 1.111325         # [-] efficiency of specific exhaust velocity
    #                 }

    # # Parameters of calculation condition
    # PARAM_CALCOND = {"dt": 0.001,       # [s] time resolution
    #                  "dx": 0.1e-4,      # [m] space resolution
    #                 #  "dx": 0.1e-3,      # [m] space resolution
    #                 #  "t_end": 1.0,     # [s] end time of calculation
    #                  "t_end": 30.85,     # [s] end time of calculation
    #                  "x_max": 15.0e-3,   # [m] maximum calculation region
    #                 #  "x_max": 10.0e-3,   # [m] maximum calculation region
    #                  "Vf_mode": False,  # whether calculation uses Vf (radial integretion) or not in mf calculation
    #                  "Af_modify": True  # whether modify the effect of 2D idealization for burning area, Af, or not.
    #                 }

    # # Constant of fuel regression model and experimental regression rate formula
    # PARAM_MODELCONST = {"Cr": 20.0e-6,  # regressoin constant that reflects the effect of combustion gas visocosity and blowing number
    #                     # "Cr": 3.01e-6,  # regressoin constant that reflects the effect of combustion gas visocosity and blowing number
    #                     # "Cr": 4.58e-6,  # regressoin constant that reflects the effect of combustion gas visocosity and blowing number
    #                     "z": 0.6,       # exponent constant of propellant mass flux, G.
    #                     "m": -0.2,      # exponent constant of distance from leading edge of fuel, x.
    #                     "k": 3.0e+4,    # experimental constant, which multiply on G when calculate theta, which reflect the effect of leading edge of boundary layer 
    #                     "C1": 1.39e-7,  # experimental constant of experimental regression rate formula
    #                     "C2": 1.61e-9,  # experimental constant of experimental regression rate formula
    #                     "n": 1.0        # experimental exponent constant of pressure
    #                     }
    
    # # Function list of NASA-CEA calculation result
    # CEA_FLDPATH = os.path.join("cea_db", "GOX_CurableResin", "csv_database")        # folder path, which contain csv data-base of CEA results
    # FUNCLIST_CEA = {"func_T": Read_datset(CEA_FLDPATH).gen_func("T_c"),             # gas temeratur interporate function
    #                 "func_CSTAR": Read_datset(CEA_FLDPATH).gen_func("CSTAR"),       # c* interporate function
    #                 "func_M": Read_datset(CEA_FLDPATH).gen_func("M_c"),             # molecular weight interpolate function
    #                 "func_cp": Read_datset(CEA_FLDPATH).gen_func("Cp_c"),           # spcific heat interpolate function
    #                 "func_gamma": Read_datset(CEA_FLDPATH).gen_func("GAMMAs_c")     # specific heat ratio interpolate function
    #                 }

    # PARAM_PLOT = {"interval": 0.1,     # [s] plot interval of movie
    #               "y_max": 5.0e-3,      # [m] plot width of fuel regression shape
    #              }

    # def FUNC_MOX(t):
    #     """Function of oxidizer mass flow rate [kg/s]
        
    #     Parameters
    #     ----------
    #     t : float
    #         time [s]
        
    #     Returns
    #     -------
    #     mox: float
    #         oxidizer mass flow rate [kg/s]
    #     """
    #     mox1 = 2.0e-3 # [kg/s] oxidizer mass flow rate before slottling
    #     mox2 = 4.5e-3 # [kg/s] oxidizer mass flow rate after slottling
    #     if t < 5.0:
    #         mox = mox1
    #     elif 5.0<=t and t<14.0:
    #         mox = mox2
    #     elif 14.0<=t and t<20.0:
    #         mox = mox1
    #     else:
    #         mox = mox2
    #     return mox

    ###### The Above Section is to Conduct Debugging ######

# %% Read the calculation condition from cond.py and mox.csv
    FLDNAME, COND, FUNC_MOX = read_cond()
    PARAM_EXCOND = COND["PARAM_EXCOND"]
    PARAM_CALCOND = COND["PARAM_CALCOND"]
    PARAM_MODELCONST = COND["PARAM_MODELCONST"]
    CEA_FLDPATH = COND["CEA_FLDPATH"]
    FUNCLIST_CEA = COND["FUNCLIST_CEA"]
    PARAM_PLOT = COND["PARAM_PLOT"]

# %%  Generate instance of simulation, excete calculation and output all of the results
    inst = Main(PARAM_EXCOND, PARAM_CALCOND, PARAM_MODELCONST, CEA_FLDPATH,FUNCLIST_CEA, PARAM_PLOT, fldname=FLDNAME)
    FIG = plt.figure(figsize=(37,20))
    inst.exe(FUNC_MOX)
    print("CFL = {}".format(inst.cond_cal["CFL"]))
    DIC_RESULT  = {"t": inst.t_history,
                  "Pc": inst.Pc_history,
                  "Vox": inst.Vox_history,
                  "Vf": inst.Vf_history,
                  "mox": inst.mox_history,
                  "mf": inst.mf_history,
                  "of": inst.of_history,
                  "cstr": inst.cstr_history,
                  "Re": inst.Re_history,
                  "ustr_lam": inst.ustr_lam_history,
                  "ustr_turb": inst.ustr_turb_history
                  }
    DF_RESULT = pd.DataFrame(DIC_RESULT)
    DF_RESULT.to_csv(os.path.join(inst.fld_name, "result.csv"))
    DF_R_RESULT = pd.DataFrame(inst.r_history, index=inst.t_history, columns=inst.x)
    DF_R_RESULT.to_csv(os.path.join(inst.fld_name, "result_r.csv"))
    DF_RDOT_RESULT = pd.DataFrame(inst.rdot_history, index=inst.t_history, columns=inst.x)
    DF_RDOT_RESULT.to_csv(os.path.join(inst.fld_name, "result_rdot.csv"))

# %% Generate animation.
    inst.gen_img_list(FIG)
    print("Now generating animation...")
    anim = ArtistAnimation(FIG, inst.img_list, interval=PARAM_CALCOND["dt"]*1e+3)
    anim.save(os.path.join(inst.fld_name, "animation.mp4"), writer="ffmpeg", fps=1/PARAM_PLOT["interval"])
    print("Completed!")