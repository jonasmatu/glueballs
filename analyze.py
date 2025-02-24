import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
import h5py

from scipy import optimize
from scipy import interpolate

import calcBounce as cb
from calc_nucleation import findNucleationTemp
from potential import Potential


plt.style.use("plots.mplstyle")

def plotTnuc(fname):
    f = h5py.File(fname, "r")
    data = f["data"]
    npoints = data.attrs["npoints"]
    virstart, virend = data.attrs["vir_start"], data.attrs["vir_end"]
    epsstart, epsend = data.attrs["eps_start"], data.attrs["eps_end"]
    xmin = data.attrs["xmin"] 
    deltat = data.attrs["deltat"]
    n = data.attrs["n"]
    N = data.attrs["N"]
    withQCD = data.attrs["withQCD"]
    Tnuc3 = data["data/Tnuc3"][:,:]
    Tnuc4 = data["data/Tnuc4"][:,:]
    f.close()

    vir_range = np.linspace(virstart, virend, npoints)
    eps_range = np.linspace(epsstart, epsend, npoints)

    fig, ax = plt.subplots(1,2,figsize=(12,6),layout="constrained", sharey=True)

    Tnuc3[np.where(Tnuc3 <= 0)] = np.nan

    vmin = np.nanmin(Tnuc3)
    vmax = np.nanmax(Tnuc3)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    abc = ax[0].pcolor(eps_range, vir_range, Tnuc3, cmap="viridis", norm=norm)
    clb = plt.colorbar(abc, ax=ax[0], norm=norm)
    clb.ax.set_title(r"$T_\mathrm{nuc}$")

    Tnuc4[np.where(Tnuc4 <= 0)] = np.nan
    vmin = np.nanmin(Tnuc4)
    vmax = np.nanmax(Tnuc4)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    
    abc = ax[1].pcolor(eps_range, vir_range, Tnuc4, cmap="viridis", norm=norm)
    clb = plt.colorbar(abc, ax=ax[1], norm=norm)
    clb.ax.set_title(r"$T_\mathrm{nuc}$")

    ax[0].set_ylabel(r"$v_{\mathrm{IR}}$")
    ax[0].set_xlabel(r"$\epsilon$")
    ax[1].set_xlabel(r"$\epsilon$")

    ax[0].set_title(r"O(3) tunneling")
    ax[1].set_title(r"O(4) tunneling")
    plt.suptitle(f"n = {n:}, " + r"$\tilde{\delta} = $" + f"{deltat:}, " \
                 + r"$\mu_\mathrm{min}$ = " + f"{xmin:3.0f}", fontsize=14)
    

    plt.show()

