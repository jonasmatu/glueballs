"""Module to test stuff

Author: Jonas Matuszak <jonas.matuszak@kit.edu>, 2025
"""
from calcBounce import *
from potential import Potential

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
                                      options={"xatol": 1e-20}).x
    phimeta = optimize.minimize_scalar(V, bounds=(-T, phibar),
                                       options={"xatol": 1e-20}).x
    # Sometimes only finds local minimum
    if V(phimeta) > V(-T):
        phimeta = -T
    phiroot = optimize.brentq(lambda phi: V(phi) - V(phimeta), phibar, xmin-1e-15)
    if debug:
        print("phiroot = ", phiroot)
        print("V(phiroot) - V(-T) = ", V(phiroot)- V(-T))
        print("phibar = ", phibar)
        print("V(phibar)", V(phibar))
        print("phimeta = ", phimeta)
        print("V(phimeta)", V(phimeta))
        print("at phibar = ", V(phibar) - V(phimeta))
        print("at xmin = ", V(xmin) - V(phimeta))
        pot.plotPotentialAll(phiroot*1.1, T, -T)
        
    

    # print("phibar = ", phibar)
    # print("phiroot = ", phiroot)

    insta = Instanton(V, dV, d2V, ndim=ndim, Nkin=3*N**2/(2*np.pi**2),
                      xtol=1e-15, phitol=1e-12, rmin_scaling=1e-8) # This setting is very important for small T!
    r, y, ctype = insta.findProfile(phimeta, xmin, phibar, f=1e-3)
    phi, dphi = y.T
    phi0 = phi[0]
    S = insta.calcAction(r, phi, dphi, phimeta)
    if debug:
        print("ctype = ", ctype)
        print(f"S = {S:2.5g}")
        print(f"phi0 = {phi0:2.5g}")
        fig, ax = plt.subplots(1,1,layout="constrained")
        plt.semilogx(r, phi)
        plt.xlabel("r")
        plt.ylabel(r"$\phi$")
        plt.show()
    
    
    return S, phi0




def scan_n_at_T(T, fname, nrange=(0.1,0.7), npoints=50):
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

    # nrange = np.logspace(np.log10(0.1), np.log10(0.7), npoints)
    nrange = np.linspace(nrange[0], nrange[1], npoints)
    S4range = np.zeros_like(nrange)
    S4aprrange = np.zeros_like(nrange)
    S4TRrange = np.zeros_like(nrange)
    S3range = np.zeros_like(nrange)
    S3aprrange = np.zeros_like(nrange)
    S3TRrange = np.zeros_like(nrange)
    for i, n in enumerate(nrange):
        try:
            S4range[i] ,_ = getActionAtT(T, xmin, vir, eps, delta, n, N, withQCD=True, ndim=4)
            pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=True)
            S4aprrange[i], _ = S4TriangleApprox(pot, T)
            S4TRrange[i], _ = triangleApproxAction(pot, T, 4)
        except Exception as e:
            print(e)
            S4range[i] = np.nan
        try:
            S3range[i] ,_ = getActionAtT(T, xmin, vir, eps, delta, n, N, withQCD=True, ndim=3)
            pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=True)
            S3aprrange[i], _ = S3TriangleApprox(pot, T)
            S3TRrange[i], _ = triangleApproxAction(pot, T, 3)
        except Exception as e:
            print(e)
            S3range[i] = np.nan

    header = f"T={T:}, xmin={xmin:}, vir={vir:}, eps={eps:}, delta={delta:}, N={N:}"
    np.savetxt(fname, np.asarray([nrange, S4range, S4aprrange,
                                  S4TRrange, S3range, S3aprrange, S3TRrange]), header=header, delimiter=",")
    

def debug_n_at_T(T, fname, nrange=(0.1,0.7), npoints=50):
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
    withQCD=True
    ndim=4

    # nrange = np.logspace(np.log10(0.1), np.log10(0.7), npoints)
    nrange = np.linspace(nrange[0], nrange[1], npoints)
    S4range = np.zeros_like(nrange)
    S3range = np.zeros_like(nrange)


    rrange = []
    phirange = []
    Vrange = []
    Vphirange = []
    Srange = []
    phimeta_range = []
    
    for i, n in enumerate(nrange):
        try:
            pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=withQCD)

            V = lambda x: pot.Vfull(x, T)
            dV = lambda x: pot.dVfull(x, T)
            d2V = lambda x: pot.d2Vfull(x, T)
    
            phibar = optimize.minimize_scalar(lambda phi: -V(phi), bounds=(0, xmin),
                                      options={"xatol": 1e-15}).x
            phimeta = optimize.minimize_scalar(V, bounds=(-T*1.01, phibar),
                                       options={"xatol": 1e-15}).x
            if V(phimeta) > V(-T):
                phimeta = -T
            phimeta_range.append(phimeta)
            phiroot = optimize.brentq(lambda phi: V(phi) - V(phimeta), phibar, xmin-1e-15)
                
            # print("V(phibar)", V(phibar))
            # print("phimeta = ", phimeta)
            # print("V(phimeta)", V(phimeta))
            # print("at phibar = ", V(phibar) - V(phimeta))
            # print("at xmin = ", V(xmin) - V(phimeta))
            # pot.plotPotentialAll(phiroot*1.1, T, phimeta)
            insta = Instanton(V, dV, d2V, ndim=ndim, Nkin=3*N**2/(2*np.pi**2),
                              xtol=1e-15, phitol=1e-12, rmin_scaling=1e-6) # This setting is very important for small T!
            r, y, ctype = insta.findProfile(phimeta, xmin, phibar)
            phi, dphi = y.T
            phi0 = phi[0]
            S = insta.calcAction(r, phi, dphi, phimeta)
            phirange.append(phi)
            rrange.append(r)
            Srange.append(S)
            Vphirange.append(np.linspace(-T, phiroot, 100))
            Vrange.append(V(Vphirange[i]))

            # plt.semilogx(r, phi, color=cols[i])

        except Exception as e:
            print(e)
            S4range[i] = np.nan

    cmap = plt.get_cmap("plasma")
    cols = cmap.resampled(256)
    cols = cmap(np.linspace(0, 1, npoints))


    fig, ax = plt.subplots(1,3, layout="constrained")
    for i in range(len(rrange)):
        ax[0].plot(nrange[i], Srange[i], 'o', color=cols[i])
        ax[1].axhline(phimeta_range[i], color=cols[i], alpha=0.5)
        ax[1].semilogx(rrange[i], phirange[i], color=cols[i])
        ax[2].plot(Vphirange[i], Vrange[i], color=cols[i])
    norm = mpl.colors.Normalize(vmin=nrange[0], vmax=nrange[-1])
    clb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    clb.ax.set_ylabel(r"$n$")
    ax[0].set_xlabel(r"$n$")
    ax[0].set_ylabel(r"$S_4$")
    ax[1].set_xlabel(r"$r$")
    ax[1].set_ylabel(r"$\phi$")
    ax[2].set_xlabel(r"$\phi$")
    ax[2].set_ylabel(r"$V(\phi, T)$")
    plt.show()

    

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

    fig, ax = plt.subplots(1,1, layout="constrained")
    try:
        nrange, S4range, S4aprrange, S4TRrange, S3range, S3aprrange, S3TRrange = np.loadtxt(fname, delimiter=",")
        # plt.semilogy(nrange, S4aprrange, label=r"$S_4$ Servant&Harling", color="C1", ls="--")
        plt.semilogy(nrange, S4TRrange, label=r"$S_4$ triangle", color="C1", ls="-.")
        if T != 0:
            # plt.semilogy(nrange, S3aprrange/T, label=r"$S_3/T$ Servant&Harling", color="C0", ls="--")
            plt.semilogy(nrange, S3TRrange/T, label=r"$S_3/T$  triangle", color="C0", ls="-.")
    except:
        nrange, S4range, S3range = np.loadtxt(fname, delimiter=",")
    plt.semilogy(nrange, S4range, label=r"$S_4$", color="C1")
    if T != 0:
        plt.semilogy(nrange, S3range/T, label=r"$S_3/T$", color="C0")
    plt.title(f"T = {T:2.3g}, mumin = {xmin:2.1g}, vir = {vir:2.2g},\n" + \
              f"epsilon = {eps:2.3g}, delta = " + f"{delta:2.1g}, " + \
              f"N = {N:2.2g}")
    # if T != 0.0:
    #     plt.ylim(0, 1.1*np.maximum(np.max(S3range/T), np.max(S4range)))
    plt.ylabel("Action")
    plt.xlabel("n")
    plt.legend()
    plt.show()
