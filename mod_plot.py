# coding: utf-8

"""
Content: Module for generate the list of plot
Author: Ayumu Tsuji @Hokkaido University

Description:
Generate the list of plot which is used for making animation file.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from tqdm import tqdm

matplotlib.style.use("tj_origin.mplstyle")

def _plot_(t, dic_axis, dic_dat, **cond):
    """ Function for plotting the simulated history
    
    Parameters
    ----------
    t : float
        time [s]
    dic_axis : dic of Axes objects of matplotlib
        dictionary of Axes objects defined by matplotlib
    dic_dat: dict of 1D-ndarray
        dictonary of calculation result array
    cond: dict
        dictonary of calculatioin, plot and experimental condition
    
    Returns
    -------
    img1 : Line2D object of matplotlib
        Line2D object of fuel regression shape history 
    img2 : Line2D object of matplotlib
        Line2D object of pressure and oxidizer port velocity history
    img3 : Line2D object of matplotlib
        Line2D object of oxidizer mass flow rate and fuel mass flow rate history
    img4 : Line2D object of matplotlib
        Line2D object of regression shape and radial regression rate history
    img5 : Line2D object of matplotlib
        Line2D object of c* and O/F history
    img6 : Line2D object of matplotlib
        Line2D object of axial regression rate history
    """
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

    x = dic_dat["x"]
    pitch = cond["pitch"]
    x_front = np.sort(-x)
    d = cond["d"]
    index = np.where(dic_dat["t_history"] == t)[0][0]
    t_history = dic_dat["t_history"][:(index+1)]
    
    ## plot part of regression shape, r.
    r = dic_dat["r_history"][index]
    r_front = np.array([0.0 for i in r])
    img1 = ax1.plot(np.append(x_front, x)*1.0e+3, (np.append(r_front, r)+d/2)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, -(np.append(r_front, r)+d/2)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, (np.append(r_front, r)+d/2 +pitch)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, -(np.append(r_front, r)+d/2 +pitch)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, (np.append(r_front, r)+d/2 -pitch)*1.0e+3, color="b")\
            + ax1.plot(np.append(x_front, x)*1.0e+3, -(np.append(r_front, r)+d/2 -pitch)*1.0e+3, color="b")

    ## plot part of chamber pressure Pc and oxidizer port velocity Vox.
    Pc_history = dic_dat["Pc_history"][:(index+1)]
    Vox_history = dic_dat["Vox_history"][:(index+1)]
    img2 = ax2_sub.plot(t_history, Vox_history, color="b", label="$V_{ox}$")\
            + ax2.plot(t_history, Pc_history*1.0e-6, color="r", label="$P_c$")

    ## plot part of fuel and oxidizer mass flow rate, mf and mox.
    mf_history = dic_dat["mf_history"][:(index+1)]
    mox_history = dic_dat["mox_history"][:(index+1)]
    img3 = ax3.plot(t_history, mf_history*1.0e+3, color="r", label="$\dot m_f$")\
            + ax3_sub.plot(t_history, mox_history*1.0e+3, color="b", label="$m_{ox}$")

    # plot part of regression shape r and radial regression rate rdot.
    rdot = dic_dat["rdot_history"][index]
    img4 = ax4_sub.plot(x*1.0e+3, rdot*1.0e+3, color="r", label=r"$\dot r$") + ax4.plot(x*1.0e+3, r*1.0e+3, color="b", label=r"$r$")

    ## plot part of characteristic exhaust velocity c* and of ratio O/F.
    cstr_history = dic_dat["cstr_history"][:(index+1)]
    of_history = dic_dat["of_history"][:(index+1)]
    img5 = ax5_sub.plot(t_history, of_history, color="b", label=r"$O/F$") + ax5.plot(t_history, cstr_history, color="r", label=r"$c^*$")

    ## plot part of axial regression rate Vf.
    Vf_history = dic_dat["Vf_history"][:(index+1)]
    img6 = ax6.plot(t_history, Vf_history*1.0e+3, color="r")
    return img1, img2, img3, img4, img5, img6


def gen_img_list(fig, img_list, dic_dat, **cond):
    """ Stuckking the image file to img_list
    
    Parameters
    ----------
    fig : Figure object of matplotlib
    self.img_list: list of Figure objects of matplotlib
        image list stacked new plot image
    dic_dat: dict of 1D-ndarray
        dictonary of calculation result array
    cond: dict
        dictonary of calculatioin, plot and experimental condition
                    
    Return
    ----------
    img_list: list of Figure objects of matplotlib
        image list stacked new plot image
    """
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    fig.subplots_adjust(wspace=0.4)
    fig.subplots_adjust(hspace=0.2)
    fig.subplots_adjust(bottom=0.075)
    fig.subplots_adjust(top=0.925) 
    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(left=0.05)

    if "intrv" in cond:
        intrv = cond["intrv"]
    else:
        intrv = cond["interval"]
    x_max = cond["x_max"]
    dt = cond["dt"]
    t_end = cond["t_end"]

    ## Regression shape
    ax1.set_xlabel("Axial distance $x$ [mm]")
    ax1.set_ylabel("Regression shape [mm]")
    x_begin = -x_max/4*1.0e+3
    x_end = x_max*1.0e+3
    y_begin = -(x_end-x_begin)/2
    y_end = (x_end-x_begin)/2
    ax1.set_ylim(y_begin, y_end)
    ax1.set_xlim(x_begin, x_end)
    ax1.grid()
    
    ## Chamber pressure v.s. t
    ax2.set_xlabel("Time $t$ [s]")
    ax2.set_ylabel("Chamber pressure $P_c$ [MPa]")
    ax2.set_xlim(0, t_end)
    ax2.set_ylim(0, 1.1 *dic_dat["Pc_history"].max()*1e-6)
    ax2.grid()
    ax2_sub = ax2.twinx()
    ax2_sub.set_ylabel("Oxidizer port velocity $V_{ox}$ [m/s]")
    ax2_sub.set_ylim(0, 1.1 *dic_dat["Vox_history"].max())
    
    ## Fuel and oxidizer mass flow rate v.s. t
    ax3.set_xlabel("Time $t$ [s]")
    ax3.set_ylabel("Fuel mass flow rate $\dot m_f$ [g/s]")
    ax3.set_xlim(0, t_end)
    ax3.set_ylim(0, 1.2 *dic_dat["mf_history"].max()*1e+3)
    ax3.grid()
    ax3_sub = ax3.twinx()
    ax3_sub.set_ylabel("Oxidizer mass flow rate $\dot m_{ox}}}$ [g/s]")
    ax3_sub.set_ylim(0, 1.2 *dic_dat["mox_history"].max()*1e+3)

    ## radial regression distance  r v.s. t radial regression rate  rdot v.s. t
    ax4.set_xlabel("Axial distance $x$ [mm]")
    ax4.set_ylabel("Radial regression distance $r$ [mm]")
    max_index = 0
    for r_array in dic_dat["r_history"]:
        tmp = r_array[~np.isnan(r_array)].size - 1
        if tmp > max_index:
            max_index = tmp
    ax4.set_xlim(0, 1.1*dic_dat["x"][max_index]*1.0e+3)
    ax4.set_ylim(0.0, 1.1*(cond["pitch"]-cond["d"])/2*1e+3)
    ax4.grid()
    ax4_sub = ax4.twinx()
    ax4_sub.set_ylabel("Radial regression rate $\dot r$ [mm/s]")
    ax4_sub.set_ylim(0.0, dic_dat["rdot_history"][~np.isnan(dic_dat["rdot_history"])].max()*1e+3)
    
    ## specific exhaust velocity and of ratio v.s. t
    ax5.set_xlabel("Time $t$ [s]")
    ax5.set_ylabel("Characteristic exhaust velocity $c^*$ [m/s]")
    ax5.set_xlim(0, t_end)
    ax5.set_ylim(0.9*dic_dat["cstr_history"].min(), 1.2 *dic_dat["cstr_history"].max())
    ax5.grid()
    ax5_sub = ax5.twinx()
    ax5_sub.set_ylabel("Oxidizer to fuel mass flow ratio $O/F$ [-]")
    ax5_sub.set_ylim(0.9*dic_dat["of_history"][~np.isnan(dic_dat["of_history"])].min(), 1.1*dic_dat["of_history"][~np.isnan(dic_dat["of_history"])].max())
    
    ## Acial regression rate v.s. t
    ax6.set_xlabel("Time $t$ [s]")
    ax6.set_ylabel("Axial regression rate $V_f$ [mm/s]")
    ax6.set_xlim(0, t_end)
    ax6.set_ylim(0, 1.1 *dic_dat["Vf_history"].max()*1e+3)
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

    ## generate a list of figures for making animation at the following iteration.
    print("Make list of image file for animation")
    for t in tqdm(dic_dat["t_history"]):
        if int(t/dt) % int(intrv/dt) == 0 or t==0.0:
            title = ax1.text(x_end/2, y_end*1.1, "t={} s".format(round(t,3)), fontsize="large")
            img1, img2, img3, img4 ,img5, img6 = _plot_(t, dic_axis, dic_dat, **cond)
            img_list.append(img1 + img2 + img3 + img4 + img5 + img6 + [title])
    
    ## add legend
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
    return img_list