# -*- coding: utf-8 -*-
# %%
import os
import sys
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
    def __init__(self, fld_name):
        if os.path.exists(fld_name):
            self.fld_name = fld_name
        else:
            print("There is not the following folder; \"{}\"".format(fld_name))
            sys.exit()
        self.df, self.df_r, self.df_rdot = self._read_dat_()
        self._initialize_()

    def _read_dat_(self):
        fpath_cond = os.path.join(self.fld_name, "cond.json")
        fpath_result = os.path.join(self.fld_name, "result.csv")
        fpath_r = os.path.join(self.fld_name, "result_r.csv")
        fpath_rdot = os.path.join(self.fld_name, "result_rdot.csv")
        if os.path.isfile(fpath_cond):
            with open(fpath_cond, mode="r") as fcond:
                cond = json.load(fcond)
                self.cond_ex = cond["PARAM_EXCOND"]
                self.cond_cal = cond["PARAM_CALCOND"]
                self.const_model = cond["PARAM_MODELCONST"]
        else:
            print("There is not the file named as \"{}\"".format(fpath_cond))
        if os.path.isfile(fpath_result):
            df = pd.read_csv(fpath_result, header=0)
            self.t_history = np.array(df.t)
            self.Pc_history = np.array(df.Pc)
            self.Vox_history = np.array(df.Vox)
            self.Vf_history = np.array(df.Vf)
            self.mox_history = np.array(df.mox)
            self.mf_history = np.array(df.mf)
            self.of_history = np.array(df.of)
            self.cstr_history = np.array(df.cstr)
        else:
            print("There is not the file named as \"{}\"".format(fpath_result))
        if os.path.isfile(fpath_r):
            df_r = pd.read_csv(fpath_r, index_col=0, na_filter=True)
            self.r_history = np.array(df_r)
        else:
            print("There is not the file named as \"{}\"".format(fpath_r))
        if os.path.isfile(fpath_rdot):
            df_rdot = pd.read_csv(fpath_rdot, index_col=0, na_filter=True)
            self.rdot_history = np.array(df_rdot)
        else:
            print("There is not the file named as \"{}\"".format(fpath_rdot))
        return df, df_r, df_rdot


    def _initialize_(self):
        """ Initialize some class variable
        """
        self.x = np.arange(0.0, self.cond_cal["x_max"]+self.cond_cal["dx"]/2, self.cond_cal["dx"])
        self.t_history = np.arange(0.0, self.cond_cal["t_end"]+self.cond_cal["dt"]/2, self.cond_cal["dt"])
        self.img_list = []  # define the list in wihch images of plot figure are stacked


    def _plot_(self, t, dic_axis):
        ax1 = dic_axis["ax1"]
        ax2 = dic_axis["ax2"]
        ax2_sub = dic_axis["ax2_sub"]
        ax3 = dic_axis["ax3"]
        ax3_sub = dic_axis["ax3_sub"]
        ax4 = dic_axis["ax4"]
        ax4_sub = dic_axis["ax4_sub"]
        ax5 = dic_axis["ax5"]
        ax5_sub = dic_axis["ax5_sub"]
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
        img2 = ax2_sub.plot(t_history, Vox_history, color="b", label="$V_{ox}$")\
                + ax2.plot(t_history, Pc_history*1.0e-6, color="r", label="$P_c$")

        mf_history = self.mf_history[:(index+1)]
        mox_history = self.mox_history[:(index+1)]
        img3 = ax3.plot(t_history, mf_history*1.0e+3, color="r", label="$\dot m_f$")\
                + ax3_sub.plot(t_history, mox_history*1.0e+3, color="b", label="$m_{ox}$")

        rdot = self.rdot_history[index]
        img4 = ax4_sub.plot(x*1.0e+3, rdot*1.0e+3, color="r", label=r"$\dot r$") + ax4.plot(x*1.0e+3, r*1.0e+3, color="b", label=r"$r$")

        cstr_history = self.cstr_history[:(index+1)]
        of_history = self.of_history[:(index+1)]
        img5 = ax5_sub.plot(t_history, of_history, color="b", label=r"$O/F$") + ax5.plot(t_history, cstr_history, color="r", label=r"$c^*$")

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

        if "intrv" in kwargs:
            intrv = kwargs["intrv"]
        else:
            intrv = self.plot_param["interval"]
        x_max = self.cond_cal["x_max"]
        # y_max = self.plot_param["y_max"]
        dt = self.cond_cal["dt"]
        t_end = self.cond_cal["t_end"]

        # Regression shape
        ax1.set_xlabel("Axial distance $x$ [mm]")
        ax1.set_ylabel("Regression shape [mm]")
        x_begin = -x_max/4*1.0e+3
        x_end = x_max*1.0e+3
        y_begin = -(x_end-x_begin)/2
        y_end = (x_end-x_begin)/2
        ax1.set_ylim(y_begin, y_end)
        ax1.set_xlim(x_begin, x_end)
        ax1.grid()
        
        # Chamber pressure v.s. t
        ax2.set_xlabel("Time $t$ [s]")
        ax2.set_ylabel("Chamber pressure $P_c$ [MPa]")
        ax2.set_xlim(0, t_end)
        ax2.set_ylim(0, 1.1 *self.Pc_history.max()*1e-6)
        ax2.grid()
        ax2_sub = ax2.twinx()
        ax2_sub.set_ylabel("Oxidizer port velocity $V_{ox}$ [m/s]")
        ax2_sub.set_ylim(0, 1.1 *self.Vox_history.max())
        
        # Fuel and oxidizer mass flow rate v.s. t
        ax3.set_xlabel("Time $t$ [s]")
        ax3.set_ylabel("Fuel mass flow rate $\dot m_f$ [g/s]")
        ax3.set_xlim(0, t_end)
        ax3.set_ylim(0, 1.2 *self.mf_history.max()*1e+3)
        ax3.grid()
        ax3_sub = ax3.twinx()
        ax3_sub.set_ylabel("Oxidizer mass flow rate $\dot m_{ox}}}$ [g/s]")
        ax3_sub.set_ylim(0, 1.2 *self.mox_history.max()*1e+3)

        # radial regression distance  r v.s. t radial regression rate  rdot v.s. t
        ax4.set_xlabel("Axial distance $x$ [mm]")
        ax4.set_ylabel("Radial regression distance $r$ [mm]")
        max_index = 0
        for r_array in self.r_history:
            tmp = r_array[~np.isnan(r_array)].size - 1
            if tmp > max_index:
                max_index = tmp
        ax4.set_xlim(0, 1.1*self.x[max_index]*1.0e+3)
        ax4.set_ylim(0.0, 1.1*(self.cond_ex["pitch"]-self.cond_ex["d"])/2*1e+3)
        ax4.grid()
        ax4_sub = ax4.twinx()
        ax4_sub.set_ylabel("Radial regression rate $\dot r$ [mm/s]")
        ax4_sub.set_ylim(0.0, self.rdot_history[~np.isnan(self.rdot_history)].max()*1e+3)
        
        # specific exhaust velocity and of ratio v.s. t
        ax5.set_xlabel("Time $t$ [s]")
        ax5.set_ylabel("Characteristic exhaust velocity $c^*$ [m/s]")
        ax5.set_xlim(0, t_end)
        ax5.set_ylim(0.9*self.cstr_history.min(), 1.2 *self.cstr_history.max())
        ax5.grid()
        ax5_sub = ax5.twinx()
        ax5_sub.set_ylabel("Oxidizer to fuel mass flow ratio $O/F$ [-]")
        ax5_sub.set_ylim(0.9*self.of_history[~np.isnan(self.of_history)].min(), 1.1*self.of_history[~np.isnan(self.of_history)].max())
        
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
                    "ax4_sub": ax4_sub,
                    "ax5": ax5,
                    "ax5_sub": ax5_sub,
                    "ax6": ax6
                    }
        print("Make list of image file for animation")
        for t in tqdm(self.t_history):
            if int(t/dt) % int(intrv/dt) == 0 or t==0.0:
                title = ax1.text(x_end/2, y_end*1.1, "t={} s".format(round(t,3)), fontsize="large")
                img1, img2, img3, img4 ,img5, img6 = self._plot_(t, dic_axis)
                self.img_list.append(img1 + img2 + img3 + img4 + img5 + img6 + [title])
        # add legend
        hl2, label2 = [ax2.get_legend_handles_labels()[0][0], ax2.get_legend_handles_labels()[1][0]]
        hl2_sub, label2_sub = [ax2_sub.get_legend_handles_labels()[0][0], ax2_sub.get_legend_handles_labels()[1][0]]
        ax2.legend([hl2, hl2_sub], [label2, label2_sub], loc="lower right")
        hl3, label3 = [ax3.get_legend_handles_labels()[0][0], ax3.get_legend_handles_labels()[1][0]]
        hl3_sub, label3_sub = [ax3_sub.get_legend_handles_labels()[0][0], ax3_sub.get_legend_handles_labels()[1][0]]
        ax3.legend([hl3, hl3_sub], [label3, label3_sub], loc="lower right")
        hl4, label4 = [ax4.get_legend_handles_labels()[0][0], ax4.get_legend_handles_labels()[1][0]]
        hl4_sub, label4_sub = [ax4_sub.get_legend_handles_labels()[0][0], ax4_sub.get_legend_handles_labels()[1][0]]
        ax4.legend([hl4, hl4_sub], [label4, label4_sub], loc="lower right")
        hl5, label5 = [ax5.get_legend_handles_labels()[0][0], ax5.get_legend_handles_labels()[1][0]]
        hl5_sub, label5_sub = [ax5_sub.get_legend_handles_labels()[0][0], ax5_sub.get_legend_handles_labels()[1][0]]
        ax5.legend([hl5, hl5_sub], [label5, label5_sub], loc="lower right")
        return self.img_list


# %%
if __name__ == "__main__":
    # import matplotlib
    # matplotlib.style.use("tj_origin.mplstyle")
# %%
    # FLD_NAME = "2020_0721_173054"
    FLD_NAME = "2020_0728_203922"
    inst = Main(FLD_NAME)

# %%
    # FIG_TMP = plt.figure(figsize=(28,16))
    # inst.gen_img_list(FIG_TMP, intrv=1.0)
# %%
    """ Part of Generating a Movie
    """
    INTERVAL = 0.1 # [s]
    FIG = plt.figure(figsize=(28,16))
    inst.gen_img_list(FIG, intrv=INTERVAL)
    animf_name = "animation_" + datetime.now().strftime("%Y_%m%d_%H%M%S") + ".mp4"
    print("Now generating animation...")
    anim = ArtistAnimation(FIG, inst.img_list, interval=inst.cond_cal["dt"]*1e+3)
    anim.save(os.path.join(inst.fld_name, animf_name), writer="ffmpeg", fps=1/INTERVAL)
    # anim.save(os.path.join(inst.fld_name, animf_name), writer="ffmpeg")
    print("Completed!")

#%%
