"""Module to test stuff

Author: Jonas Matuszak <jonas.matuszak@kit.edu>, 2025
"""
from calcBounce import *
from potential import Potential

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from scipy import optimize

plt.style.use("plots.mplstyle")

def getActionAtT(T, xmin, vir, eps, delta, n, N, withQCD, ndim, debug=False):
    """description

    Parameters
    ----------

    Returns
    ----------
    S : float
    phi0 : float
    """
    pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=withQCD)

    V = lambda x: pot.Vfull(x, T)
    dV = lambda x: pot.dVfull(x, T)
    d2V = lambda x: pot.d2Vfull(x, T)
    
    phibar = optimize.minimize_scalar(lambda phi: -V(phi), bounds=(0, xmin),
                                      options={"xatol": 1e-15}).x
    phimeta = optimize.minimize_scalar(V, bounds=(-T, phibar),
                                       options={"xatol": 1e-15}).x
    phiroot = optimize.brentq(lambda phi: V(phi) - V(phimeta), phibar, xmin-1e-15)
    if debug:
        print("phibar = ", phibar)
        print("V(phibar)", V(phibar))
        print("phimeta = ", phimeta)
        print("V(phimeta)", V(phimeta))
        print("at phibar = ", V(phibar) - V(phimeta))
        print("at xmin = ", V(xmin) - V(phimeta))
        pot.plotPotentialAll(phiroot*1.1, T, phimeta)
        
    

    # print("phibar = ", phibar)
    # print("phiroot = ", phiroot)

    insta = Instanton(V, dV, d2V, ndim=ndim, Nkin=3*N**2/(2*np.pi**2),
                      xtol=1e-15, phitol=1e-12) # This setting is very important for small T!
    r, y, ctype = insta.findProfile(phimeta, xmin, phibar)
    phi, dphi = y.T
    phi0 = phi[0]

    if debug:
        plt.semilogx(r, phi)
        plt.xlabel("r")
        plt.ylabel(r"$\phi$")
        plt.show()
    
    S = insta.calcAction(r, phi, dphi, phimeta)
    return S, phi0




def scan_n_at_T(T, fname, npoints=50):
    """description

    Parameters
    ----------

    Returns
    ----------

    """

    xmin = 2.5e3
    vir = 1
    eps = 1/20
    delta = -.5
    N = 4.5

    
    nrange = np.logspace(np.log10(0.1), np.log10(0.7), npoints)
    S4range = np.zeros_like(nrange)
    S3range = np.zeros_like(nrange)
    for i, n in enumerate(nrange):
        try:
            S4range[i] ,_ = getActionAtT(T, xmin, vir, eps, delta, n, N, withQCD=True, ndim=4)
        except Exception as e:
            print(e)
            S4range[i] = np.nan
        try:
            S3range[i] ,_ = getActionAtT(T, xmin, vir, eps, delta, n, N, withQCD=True, ndim=3)
        except Exception as e:
            print(e)
            S3range[i] = np.nan

    header = f"T={T:}, xmin={xmin:}, vir={vir:}, eps={eps:}, delta={delta:}, N={N:}"
    np.savetxt(fname, np.asarray([nrange, S4range, S3range]), header=header, delimiter=",")



def plot_n_scan(fname):
    xmin = 0
    vir = 0
    eps = 0
    delta = 0
    N = 0
    T = 0
    with open(fname, 'r') as f:
        line = f.readline()
        line = line.replace("#", "")
        data = line.split(",")
        for d in data:
            abc = d.split("=")
            if "T" in abc[0]:
                T = float(abc[-1])
            elif "xmin" in abc[0]:
                xmin = float(abc[-1])
            elif "vir" in abc[0]:
                vir = float(abc[-1])
            elif "eps" in abc[0]:
                eps = float(abc[-1])
            elif "delta" in abc[0]:
                delta = float(abc[-1])
            elif "N" in abc[0]:
                N = float(abc[-1])
    
    nrange, S4range, S3range = np.loadtxt(fname, delimiter=",")
    fig, ax = plt.subplots(1,1, layout="constrained")
    plt.plot(nrange, S4range, label=r"$S_4$")
    plt.plot(nrange, S3range/T, label=r"$S_3/T$")
    plt.title(f"T = {T:2.3g}, mumin = {xmin:2.1g}, vir = {vir:2.2g},\n" + \
              f"epsilon = {eps:2.3g}, delta = " + f"{delta:2.1g}, " + \
              f"N = {N:2.2g}")
    plt.ylim(0, 1.1*np.maximum(np.max(S3range/T), np.max(S4range)))
    plt.ylabel("Action")
    plt.xlabel("n")
    plt.legend()
    plt.show()
