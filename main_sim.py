# -*- coding: utf-8 -*-
# %%
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from tqdm import tqdm
from cea_post import Read_datset
import mod_response
import mod_shape

# %%
class Main:
    def __init__(self, cond_ex, cond_cal, const_model, funclist_cea, plot_param):
        self.cond_ex = cond_ex
        self.cond_ex["a"] = 1 -np.power(cond_ex["d"]/cond_ex["Df"], 2)*cond_ex["N"]     # [-] fuel filling rate
        self.cond_ex["Vci"] = np.pi*np.power(cond_ex["Df"], 2)/4 * cond_ex["Lci"]       # [m^3] initial chamber volume
        self.cond_ex["R_ox"] = cond_ex["Ru"]/cond_ex["M_ox"]      # [J/kg-K] oxidizer gas constant
        self.cond_cal = cond_cal
        self.cond_cal["CFL"] = np.abs(cond_cal["Vf_max"]*cond_cal["dt"]/cond_cal["dx"])      # [-] Courant number, which must be less than unity
        self.const_model = const_model
        self.funclist_cea = funclist_cea
        self.plot_param = plot_param
        self._initialize_()
        self._gen_folder_()

    def _initialize_(self):
        """ Initialize some class variable
        """
        x_max = self.cond_cal["x_max"]
        dx = self.cond_cal["dx"]
        self.fld_name = datetime.now().strftime("%Y_%m%d_%H%M%S")   # folder name which contain animation and figure of calculation result
        self.img_list = []  # define the list in wihch images of plot figure are stacked
        self.t_history = np.array([])
        self.Pc_history = np.array([])
        self.r_history = np.empty([0, int(round((x_max+dx)/dx,0))])
        self.rdot_history = np.empty([0, int(round((x_max+dx)/dx,0))])
        self.rdotn_history = np.empty([0, int(round((x_max+dx)/dx,0))])
        self.Vf_history = np.array([])
        self.Vox_history = np.array([])
        self.mf_history = np.array([])
        self.mox_history = np.array([])
        self.r_plot = np.empty([0, int(round((x_max+dx)/dx,0))])
        self.rdot_plot = np.empty([0, int(round((x_max+dx)/dx,0))])
        self.rdotn_plot = np.empty([0, int(round((x_max+dx)/dx,0))])

    def _gen_folder_(self):
        """ Generate folder which contains calculation result and make json file
        """
        os.makedirs(self.fld_name)
        dic_json = {"PARAM_EXCOND": self.cond_ex,
                    "PARAM_CALCOND": self.cond_cal,
                    "PARAM_MODELCONST": self.const_model
                    }
        with open(os.path.join(self.fld_name, "cond.json"), "w") as f:
            json.dump(dic_json, f, ensure_ascii=False, indent=4)

    def exe(self, func_mox):
        cond = dict(self.cond_ex, **self.cond_cal, **self.const_model, **self.funclist_cea, **self.plot_param)  # combine several dict of parameters
        dt = self.cond_cal["dt"]
        N = self.cond_ex["N"]
        d = self.cond_ex["d"]
        cond["time"], cond["x"], r_tmp, rdot_tmp, rdotn_tmp = mod_shape.initialize_calvalue(**cond)
        self.x = cond["x"]
        val = {}
        for t in tqdm(cond["time"]):
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
            Vf = mod_shape.func_Vf(Vox, Pc)
            self.Vf_history = np.append(self.Vf_history, Vf)
            if t != 0:
                r_tmp = r_new_tmp
                rdot_tmp = rdot_new_tmp
                rdotn_tmp = rdotn_new_tmp
            r, rdot, rdotn = mod_shape.func_rcut(r_tmp, rdot_tmp, rdotn_tmp, self.t_history, self.Vf_history, **cond)
            self.r_history = np.vstack((self.r_history, r))
            self.rdot_history = np.vstack((self.rdot_history, rdot))
            self.rdotn_history = np.vstack((self.rdotn_history, rdotn))
            if cond["Vf_mode"]:
                mf =  N *mod_shape.func_mf(r[~np.isnan(r)].size-1, r[~np.isnan(r)], rdot[~np.isnan(rdot)], Vf=Vf, **cond)
            else:
                mf =  N *mod_shape.func_mf(r[~np.isnan(r)].size-1, r[~np.isnan(r)], rdot[~np.isnan(rdot)], **cond)
            self.mf_history = np.append(self.mf_history, mf)
            # r_plot_tmp, rdot_plot_tmp, rdotn_plot_tmp = mod_shape.func_rcut(r, rdot, rdotn, self.t_history, self.Vf_history, **cond)
            # self.r_plot = np.vstack((self.r_plot, r_plot_tmp))
            # self.rdot_plot = np.vstack((self.rdot_plot, rdot_plot_tmp))
            # self.rdotn_plot = np.vstack((self.rdotn_plot, rdotn_plot_tmp))
            # calculate next step value
            val["r"] = r_tmp
            val["rdot"] = rdot_tmp
            val["rdotn"] = rdotn_tmp
            Pc_new = mod_response.exe_RK4(t, val, func_mox, self.t_history, self.Vf_history, **cond)
            r_new_tmp, rdot_new_tmp, rdotn_new_tmp = mod_shape.exe(val, **cond)


    def _plot_(self, t, dic_axis):
        ax1 = dic_axis["ax1"]
        ax2 = dic_axis["ax2"]
        ax2_sub = dic_axis["ax2_sub"]
        ax3 = dic_axis["ax3"]
        ax3_sub = dic_axis["ax3_sub"]
        ax4 = dic_axis["ax4"]
        ax5 = dic_axis["ax5"]
        ax6 = dic_axis["ax6"]

        x = self.x
        pitch = self.cond_ex["pitch"]
        x_front = np.sort(-x)
        d = self.cond_ex["d"]
        index = np.where(self.t_history == t)[0][0]
        t_history = self.t_history[:(index+1)]
        
        r = self.r_history[index]
        r_front = np.array([0.0 for i in self.r_history[index]])
        img1 = ax1.plot(np.append(x_front, x)*1.0e+3, (np.append(r_front, r)+d/2)*1.0e+3, color="b")\
                + ax1.plot(np.append(x_front, x)*1.0e+3, -(np.append(r_front, r)+d/2)*1.0e+3, color="b")\
                + ax1.plot(np.append(x_front, x)*1.0e+3, (np.append(r_front, r)+d/2 +pitch)*1.0e+3, color="b")\
                + ax1.plot(np.append(x_front, x)*1.0e+3, -(np.append(r_front, r)+d/2 +pitch)*1.0e+3, color="b")\
                + ax1.plot(np.append(x_front, x)*1.0e+3, (np.append(r_front, r)+d/2 -pitch)*1.0e+3, color="b")\
                + ax1.plot(np.append(x_front, x)*1.0e+3, -(np.append(r_front, r)+d/2 -pitch)*1.0e+3, color="b")

        Pc_history = self.Pc_history[:(index+1)]
        Vox_history = self.Vox_history[:(index+1)]
        img2 = ax2.plot(t_history, Pc_history*1.0e-6, color="r", label="$P_c$")\
                + ax2_sub.plot(t_history, Vox_history, color="b", label="$V_{ox}$")

        mf_history = self.mf_history[:(index+1)]
        mox_history = self.mox_history[:(index+1)]
        img3 = ax3.plot(t_history, mf_history*1.0e+3, color="r", label="$\dot m_f$")\
                + ax3_sub.plot(t_history, mox_history*1.0e+3, color="b", label="$m_{ox}$")

        img4 = ax4.plot(x*1.0e+3, r*1.0e+3, color="b")

        rdot = self.rdot_history[index]
        img5 = ax5.plot(x*1.0e+3, rdot*1.0e+3, color="r")

        Vf_history = self.Vf_history[:(index+1)]
        img6 = ax6.plot(t_history, Vf_history*1.0e+3, color="r")

        return img1, img2, img3, img4, img5, img6


    def gen_img_list(self, fig, **kwargs):
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
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)
        fig.subplots_adjust(wspace=0.4)

        intrv = self.plot_param["interval"]
        x_max = self.cond_cal["x_max"]
        y_max = self.plot_param["y_max"]
        dt = self.cond_cal["dt"]
        t_end = self.cond_cal["t_end"]


        # Regression shape
        ax1.set_xlabel("Axial distance $x$ [mm]")
        ax1.set_ylabel("Regression shape [mm]")
        ax1.set_ylim(-y_max*1.0e+3, y_max*1.0e+3)
        ax1.set_xlim(-x_max/4*1.0e+3,x_max*1.0e+3)
        ax1.grid()
        
        # Chamber pressure v.s. t
        ax2.set_xlabel("Time $t$ [s]")
        ax2.set_ylabel("Chamber pressure $P_c$ [MPa]")
        ax2.set_xlim(0, t_end)
        ax2.set_ylim(0, 1.1 *self.Pc_history.max()*1e-6)
        ax2.grid()
        ax2_sub = ax2.twinx()
        ax2_sub.set_xlabel("Time $t$ [s]")
        ax2_sub.set_ylabel("Oxidizer port velocity $V_{ox}$ [m/s]")
        ax2_sub.set_xlim(0, t_end)
        ax2_sub.set_ylim(0, 1.1 *self.Vox_history.max())
        hl2, label2 = ax2.get_legend_handles_labels()
        hl2_sub, label2_sub = ax2.get_legend_handles_labels()
        ax2.legend(hl2 + hl2_sub, label2 + label2_sub)
        
        # Fuel and oxidizer mass flow rate v.s. t
        ax3.set_xlabel("Time $t$ [s]")
        ax3.set_ylabel("Fuel mass flow rate $\dot m_f$ [g/s]")
        ax3.set_xlim(0, t_end)
        ax3.set_ylim(0, 1.2 *self.mf_history.max()*1e+3)
        ax3.grid()
        ax3_sub = ax3.twinx()
        ax3_sub.set_xlabel("Time $t$ [s]")
        ax3_sub.set_ylabel("Oxidizer mass flow rate $\dot m_{ox}}}$ [g/s]")
        ax3_sub.set_xlim(0, t_end)
        ax3_sub.set_ylim(0, 1.2 *self.mox_history.max()*1e+3)
        hl3, label3 = ax3.get_legend_handles_labels()
        hl3_sub, label3_sub = ax3.get_legend_handles_labels()
        ax3.legend(hl3 + hl3_sub, label3 + label3_sub)

        # radial regression distance  r v.s. t
        ax4.set_xlabel("Axial distance $x$ [mm]")
        ax4.set_ylabel("Radial regression distance $r$ [mm]")
        ax4.set_xlim(0, x_max*1.0e+3)
        # ax4.set_ylim(0.0, 1.1 *self.r_plot.max()*1e+3)
        ax4.grid()
        
        # radial regression rate  rdot v.s. t
        ax5.set_xlabel("Axial distance $x$ [mm]")
        ax5.set_ylabel("Radial regression rate $\dot r$ [mm/s]")
        ax5.set_xlim(0, x_max*1.0e+3)
        # ax5.set_ylim(0.0, 1.1 *self.rdot_plot.max()*1e+3)
        ax5.grid()
        
        # Acial regression rate v.s. t
        ax6.set_xlabel("Time $t$ [s]")
        ax6.set_ylabel("Axial regression rate $V_f$ [mm/s]")
        ax6.set_xlim(0, t_end)
        ax6.set_ylim(0, 1.1 *self.Vf_history.max()*1e+3)
        ax6.grid()

        dic_axis = {"ax1": ax1,
                    "ax2": ax2,
                    "ax2_sub": ax2_sub,
                    "ax3": ax3,
                    "ax3_sub": ax3_sub,
                    "ax4": ax4,
                    "ax5": ax5,
                    "ax6": ax6
                    }
        print("Make list of image file for animation")
        for t in tqdm(self.t_history):
            if int(t/dt) % int(intrv/dt) == 0 or t==0.0:
                title = ax1.text(x_max/2*1e+3, y_max*1.1*1e+3, "t={} s".format(round(t,3)), fontsize="large")
                img1, img2, img3, img4 ,img5, img6 = self._plot_(t, dic_axis)
                self.img_list.append(img1 + img2 + img3 + img4 + img5 + img6 + [title])
        
        return self.img_list


# %%
if __name__ == "__main__":
    # Parameters of experimental condition
    PARAM_EXCOND = {"d": 0.3e-3,        # [m] port diameter
                    "N": 433,           # [-] the number of port
                    "Df": 38e-3,        # [m] fuel outer diameter
                    "d_exit": 1.0e-3,   # [m] port exit diameter
                    "depth": 2.0e-3,    # [m] depth of expansiion region of port exit
                    "pitch": 2.0e-3,    # [m] pitch between each ports
                    "Dt": 6.2e-3,       # [m] nozzle throat diameter
                    "Lci": 20.0e-3,     # [m] initial chamber length
                    "rho_f": 1190,      # [kg/m^3] solid fuel density
                    "M_ox": 32.0e-3,    # [kg/mol] oxidizer mass per unit mole
                    "T_ox": 300,        # [K] oxidizer temperature
                    "Ru": 8.3144598,    # [J/mol-K] Universal gas constant
                    "Pci": 0.1013e+6    # [Pa] initial chamber pressure
                    }

    # Parameters of calculation condition
    PARAM_CALCOND = {"dt": 0.001,       # [s] time resolution
                     "dx": 0.1e-3,      # [m] space resolution
                     "t_end": 5.0,      # [s] end time of calculation
                     "x_max": 8.0e-3,   # [m] maximum calculation region
                     "Vf_max": 10.0e-3, # [m/s] expected maximum axial fuel regression rate
                     "Vf_mode": False   # whether code uses Vf (radial integretion) or not in mf calculation
                    }

    # Constant of fuel regression model and experimental regression rate formula
    PARAM_MODELCONST = {"Cr": 4.58e-6,  # regressoin constant that reflects the effect of combustion gas visocosity and blowing number
                        "z": 0.9,       # exponent constant of propellant mass flux, G.
                        "m": -0.2,      # exponent constant of distance from leading edge of fuel, x.
                        "k": 3.0e+4,    # experimental constant, which multiply on G when calculate theta, which reflect the effect of leading edge of boundary layer 
                        "C1": 1.39e-7,  # experimental constant of experimental regression rate formula
                        "C2": 1.61e-9,  # experimental constant of experimental regression rate formula
                        "n": 1.0        # experimental exponent constant of pressure
                        }
    
    # Function list of NASA-CEA calculation result
    CEA_FLDPATH = os.path.join("cea_db", "GOX_CurableResin", "csv_database")        # folder path, which contain csv data-base of CEA results
    FUNCLIST_CEA = {"func_T": Read_datset(CEA_FLDPATH).gen_func("T_c"),             # gas temeratur interporate function
                    "func_CSTAR": Read_datset(CEA_FLDPATH).gen_func("CSTAR"),       # c* interporate function
                    "func_M": Read_datset(CEA_FLDPATH).gen_func("M_c"),             # molecular weight interpolate function
                    "func_cp": Read_datset(CEA_FLDPATH).gen_func("cp_c"),           # spcific heat interpolate function
                    "func_gamma": Read_datset(CEA_FLDPATH).gen_func("GAMMAs_c")     # specific heat ratio interpolate function
                    }

    PARAM_PLOT = {"interval": 0.01,     # [s] plot interval of movie
                  "y_max": 5.0e-3,      # [m] plot width of fuel regression shape
                 }

    def FUNC_MOX(t):
        mox1 = 3.0e-3 # [kg/s] oxidizer mass flow rate before slottling
        mox2 = 6.0e-3 # [kg/s] oxidizer mass flow rate after slottling
        if t < 2.0:
            mox = mox1
        else:
            mox = mox2
        return mox

# %%
    inst = Main(PARAM_EXCOND, PARAM_CALCOND, PARAM_MODELCONST, FUNCLIST_CEA, PARAM_PLOT)
    FIG = plt.figure(figsize=(28,16))
    inst.exe(FUNC_MOX)
    print("CFL = {}".format(inst.cond_cal["CFL"]))
    DIC_RESULT  = {"t": inst.t_history,
                  "Pc": inst.Pc_history,
                  "Vox": inst.Vox_history,
                  "Vf": inst.Vf_history,
                  "mox": inst.mox_history,
                  "mf": inst.mf_history
                  }
    DF_RESULT = pd.DataFrame(DIC_RESULT)
    DF_RESULT.to_csv(os.path.join(inst.fld_name, "result.csv"))
    DF_R_RESULT = pd.DataFrame(inst.r_history, index=inst.t_history, columns=inst.x)
    DF_R_RESULT.to_csv(os.path.join(inst.fld_name, "result_r.csv"))
    DF_RDOT_RESULT = pd.DataFrame(inst.rdot_history, index=inst.t_history, columns=inst.x)
    DF_RDOT_RESULT.to_csv(os.path.join(inst.fld_name, "result_rdot.csv"))
# %%
    inst.gen_img_list(FIG)
    print("Now generating animation...")
    # anim = ArtistAnimation(FIG, inst.img_list, interval=PARAM_CALCOND["dt"]*1e+3)
    # anim.save(os.path.join(inst.fld_name, "animation.mp4"), writer="ffmpeg", fps=10)
    # print("Completed!")

    





