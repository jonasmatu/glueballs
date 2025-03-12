import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
import h5py

from scipy import optimize
from scipy import interpolate
from scipy import special

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
    

    plotname = fname.replace("data/", "").replace("/data.hdf5", "") + ".png"
    plt.savefig("plots/"+plotname)
    plt.show()


def calcReheatingTemp(pot: Potential, TSM: float):
    """Use energy conservation to compute the temperature
    of the SM after the FOPT.

    Parameters
    ----------
    pot:
        Radion potential
    TSM:
        Temperture of the DS and SM plasma before PT (percol)
    Returns
    ----------
    Tg:
        Glueball temperature
    """
    

    

def getRelicDensity(B: float, N: float, LGB: float) -> float:
    """Calculate the relic density of glueballs today

    Parameters
    ----------
    B:
        Temperature ratio Tg/Tsm
    N:
        SU(N)
    LGB:
        confinement scale of the glueballs

    Returns
    ----------
    Ogh2:
        Relic density of the glue balls
    """
    ODMh2 = 0.12
    Mpl_GeV = 1.220910e19
    z = 2.1 * (N**2 - 1)**(2/5.0)/N**(18/5) * B**(3/10) * (Mpl_GeV/LGB)**(3/5)
    Ogh2 = ODMh2 * 0.056 * (N**2 - 1) * (B/1e-12)**(3/4.0) * LGB * \
        1/special.lambertw(z)
    return Ogh2


def plotRelicDensity(fname):
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
