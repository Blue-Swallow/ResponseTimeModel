import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cea_post import Read_datset
# os.chdir("..")

PARAM_EXCOND = {"d": 0.3e-3,        # [m] port diameter
                "N": 433,           # [-] the number of port
                "Df": 38e-3,        # [m] fuel outer diameter
                "d_exit": 1.0e-3,   # [m] port exit diameter
                "depth": 2.0e-3,    # [m] depth of expansion region of port exit
                "pitch": 1.7e-3,    # [m] pitch between each ports
                "Dt": 5.4e-3,       # [m] nozzle throat diameter
                # "Dt": 6.5e-3,       # [m] nozzle throat diameter
                "De": 8.0e-3,       # [m] nozzle exit diamter
                "Lci": 70.0e-3,     # [m] initial chamber length
                "rho_f": 1190,      # [kg/m^3] solid fuel density
                "M_ox": 32.0e-3,    # [kg/mol] oxidizer mass per unit mole
                "T_ox": 300,        # [K] oxidizer temperature
                "mu_ox": 20.3e-6,    # [Pa-s] oxidizer viscosity
                "Ru": 8.3144598,    # [J/mol-K] Universal gas constant
                "Pci": 0.1013e+6,   # [Pa] initial chamber pressure
                "Pa": 0.1013e+6,    # [Pa] atmospheric pressure
                "eta": 0.858,       # [-] efficiency of specific exhaust velocity
                "lambda": 0.98      # [-] efficiency of nozzle
                }

# Parameters of calculation condition
PARAM_CALCOND = {"dt": 0.001,       # [s] time resolution
                    "dx": 0.1e-4,      # [m] space resolution
                #  "dx": 0.1e-3,      # [m] space resolution
                #  "t_end": 1.0,     # [s] end time of calculation
                    "t_end": 30.85,     # [s] end time of calculation
                    "x_max": 15.0e-3,   # [m] maximum calculation region
                #  "x_max": 10.0e-3,   # [m] maximum calculation region
                    "Vf_mode": False,  # whether calculation uses Vf (radial integretion) or not in mf calculation
                    "Af_modify": True,  # whether modify the effect of 2D idealization for burning area, Af, or not.
                    "use_mox_csv": True # whether use mox history in mox.csv or the following function of FUNC_MOX.
                }

# Constant of fuel regression model and experimental regression rate formula
PARAM_MODELCONST = {"Cr": 20.0e-6,  # regressoin constant that reflects the effect of combustion gas visocosity and blowing number
                    # "Cr": 3.01e-6,  # regressoin constant that reflects the effect of combustion gas visocosity and blowing number
                    # "Cr": 4.58e-6,  # regressoin constant that reflects the effect of combustion gas visocosity and blowing number
                    "z": 0.6,       # exponent constant of propellant mass flux, G.
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
                "func_cp": Read_datset(CEA_FLDPATH).gen_func("Cp_c"),           # spcific heat interpolate function
                "func_gamma": Read_datset(CEA_FLDPATH).gen_func("GAMMAs_c")     # specific heat ratio interpolate function
                }

PARAM_PLOT = {"interval": 0.1,     # [s] plot interval of movie
                "y_max": 5.0e-3,      # [m] plot width of fuel regression shape
                }


## following function of mox history is used if "use_mox_csv" is False
def FUNC_MOX(t):
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
    mox1 = 2.0e-3 # [kg/s] oxidizer mass flow rate before slottling
    mox2 = 4.5e-3 # [kg/s] oxidizer mass flow rate after slottling
    if t < 5.0:
        mox = mox1
    elif 5.0<=t and t<14.0:
        mox = mox2
    elif 14.0<=t and t<20.0:
        mox = mox1
    else:
        mox = mox2
    return mox

