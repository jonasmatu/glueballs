"""Module to scan over the radion potential and to find the
O(4) symmetric bounce action and possible tunneling. 

Author: Jonas Matuszak <jonas.matuszak@kit.edu>, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import optimize
from scipy import interpolate

import calcBounce as cb
from potential import RadionPotential


def getAction(xmin, vir, epsilon, deltat, n, N=4.5):
    """calculate the action

    Parameters
    ----------

    Returns
    ----------

    """
    try:
        delta = deltat * vir**2
        pot = RadionPotential(xmin, vir, epsilon, delta, n, N)
        r, y, ctype = cb.findProfile(pot.VGW, pot.dVGW, pot.d2VGW, xmin,
                                     pot.xmeta_min, phi0=10, N=4.5, rmin=1e-10, rmax=1e6,
                                     rscale=1)
        phi, dphi = y.T
        phi0 = phi[0]
        S = cb.findActionRadion(r, phi, dphi, N, pot.VGW, pot.xmeta_min)
        return phi0, S
    except Exception as e:
        print(e)
        return 0, 0


def getActionQCD(xmin, vir, epsilon, deltat, n, N=4.5, rscale=1):
    try:
        delta = deltat * vir**2
        pot = RadionPotential(xmin, vir, epsilon, delta, n, N, withQCD=True)
        r, y, ctype = cb.findProfile(pot.Vfull, pot.dVfull, pot.d2Vfull, xmin,
                                     pot.xmeta_min, N=4.5, rmin=1e-30, rmax=1e5)
        phi, dphi = y.T
        phi0 = phi[0]
        S = cb.findActionRadion(r, phi, dphi, N, pot.Vfull, pot.xmeta_min)
        return phi0, S
    except Exception as e:
        print(e)
        return 0, 0


def scanTunneling(npoints=30, n_jobs=6, rescan=False):
    vir_range = np.linspace(0, 1, npoints)
    eps_range = np.linspace(0, 0.1, npoints)

    import multiprocessing
    xmin = 2.5e3
    deltat = -1/2
    n = 0.15
    N = 4.5

    if rescan:
        pool = multiprocessing.Pool(processes=n_jobs)
        input_params = []
        for i, vir in enumerate(vir_range):
            for j, eps in enumerate(eps_range):
                input_params.append((xmin, vir, eps, deltat, n, N))

        results = pool.starmap(getAction, input_params)
        pool.close()
        phi0_r = np.zeros((npoints, npoints))
        S_r = np.zeros((npoints, npoints))
        for i in range(npoints):
            for j in range(npoints):
                datapoint = results[i*npoints + j]
                phi0_r[i,j], S_r[i,j] = datapoint

        np.save("scan.npy", np.array([phi0_r, S_r]))

    phi0_r, S_r = np.load("scan.npy")

    Mpl_GeV = 1.220910e19
    tun = S_r < 4 * np.log(phi0_r * Mpl_GeV/(xmin**2))


    plt.title(f"No QCD, O(4) Tunneling with: n = {n:}, " + r"$\tilde{\delta}$ = " + f"{deltat}")
    plt.pcolor(eps_range, vir_range, tun, cmap="PiYG")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$v_{\mathrm{IR}}$")
    plt.savefig("servant.png")
    plt.show()

    S_r[np.where(S_r == 0)] = np.nan
    vmin = np.nanmin(np.log10(S_r))
    vmax = np.nanmax(np.log10(S_r))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    abc = plt.pcolor(eps_range, vir_range, np.log10(S_r), cmap="viridis")
    plt.colorbar(abc, norm=norm)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$v_{\mathrm{IR}}$")
    plt.show()
    
    return phi0_r, S_r


def scanTunnelingQCD(npoints=30, n_jobs=6, rescan=False):
    vir_range = np.linspace(0, 1, npoints)
    eps_range = np.linspace(0, 0.1, npoints)

    import multiprocessing
    xmin = 2.5e3
    deltat = -1/2
    n = 0.3
    N = 4.5
    
    if rescan:
        pool = multiprocessing.Pool(processes=n_jobs)
        input_params = []
        for i, vir in enumerate(vir_range):
            for j, eps in enumerate(eps_range):
                input_params.append((xmin, vir, eps, deltat, n, N))
            
        results = pool.starmap(getActionQCD, input_params)
        pool.close()
        phi0_r = np.zeros((npoints, npoints))
        S_r = np.zeros((npoints, npoints))
        for i in range(npoints):
            for j in range(npoints):
                datapoint = results[i*npoints + j]
                phi0_r[i,j], S_r[i,j] = datapoint

        np.save("scanQCD.npy", np.array([phi0_r, S_r]))

    phi0_r, S_r = np.load("scanQCD.npy")

    phi0_r[np.where(phi0_r == 0)] = np.nan
    S_r[np.where(S_r == 0)] = np.nan

    Mpl_GeV = 1.220910e19
    tun = S_r < 4 * np.log(phi0_r * Mpl_GeV/(xmin**2))

    plt.title(f"With QCD, O(4) Tunneling with: n = {n:}, " + r"$\tilde{\delta}$ = " + f"{deltat}")
    plt.pcolor(eps_range, vir_range, tun, cmap="PiYG")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$v_{\mathrm{IR}}$")
    plt.savefig("servantQCD.png")
    plt.show()

    S_r[np.where(S_r == 0)] = np.nan
    vmin = np.nanmin(np.log10(S_r))
    vmax = np.nanmax(np.log10(S_r))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    abc = plt.pcolor(eps_range, vir_range, np.log10(S_r))
    clb = plt.colorbar(abc, norm=norm)
    clb.ax.set_ylabel(r"log$_{10}(S_4)$")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$v_{\mathrm{IR}}$")
    plt.show()


    data = []
    for i, vir in enumerate(vir_range):
        for j, eps in enumerate(eps_range):
            data.append([vir, eps, phi0_r[i,j], S_r[i,j]])

    np.savetxt("S4_actions_withQCD.txt", np.array(data), header="vir, eps, phi0, S4", delimiter=",")
    
    
    return phi0_r, S_r


def getBadPoints(Sr):
    badp = []
    n = len(Sr)
    vir_range = np.linspace(0, 1, n)
    eps_range = np.linspace(0, 0.1, n)
    for i, vir in enumerate(vir_range):
        for j, eps in enumerate(eps_range):
            if Sr[i, j] < 0:
                badp.append((vir, eps, Sr[i,j]))

    return badp

def test_badpoints():
    xmin = 2.5e3
    deltat = -1/2
    n = 0.15
    N = 4.5

    
    eps = 0.025
    vir = 0.5

    eps = 0.024137931034482762
    vir = 0.034482758620689655

    delta = deltat * vir**2
    pot = RadionPotential(xmin, vir, epsilon, delta, n, N)
    r, y, ctype = cb.findProfile(pot.VGW, pot.dVGW, pot.d2VGW, xmin,
                                     xmin*1e-4, N=4.5, rmin=1e-10, rmax=1e4, rscale=100)
    phi, dphi = y.T
    phi0 = phi[0]
    S = cb.findActionRadion(r, phi, dphi, N, pot.VGW)
    
    



