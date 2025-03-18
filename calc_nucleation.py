"""Module to calculate the nucleation temperature

Author: Jonas Matuszak <jonas.matuszak@kit.edu>, 2025
"""
import numpy as np
import matplotlib.pyplot as plt

from calcBounce import *
from potential import Potential

from scipy import optimize


def getActionAtT(T: float, pot: Potential, ndim: int, debug=False) -> (float, float):
    """description

    Parameters
    ----------

    Returns
    ----------
    S : float
    phi0 : float
    """

    V = lambda x: pot.Vfull(x, T)
    dV = lambda x: pot.dVfull(x, T)
    d2V = lambda x: pot.d2Vfull(x, T)
    
    phibar = optimize.minimize_scalar(lambda phi: -V(phi), bounds=(0, pot.xmin),
                                      options={"xatol": 1e-20}).x
    phimeta = optimize.minimize_scalar(V, bounds=(-T, phibar),
                                       options={"xatol": 1e-20}).x
    # Sometimes only finds local minimum
    if V(phimeta) > V(-T):
        phimeta = -T
    phiroot = optimize.brentq(lambda phi: V(phi) - V(phimeta), phibar, pot.xmin-1e-15)
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

    insta = Instanton(V, dV, d2V, ndim=ndim, Nkin=3*pot.N**2/(2*np.pi**2),
                      xtol=1e-14, phitol=1e-12, rmin_scaling=1e-8) # This setting is very important for small T!
    r, y, ctype = insta.findProfile(phimeta, pot.xmin, phibar, f=1e-3)
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


def nuclCriterion(S: float, phi0: float, T: float, xmin: float) -> float:
    """Nucleation criterion

    TODO: improve

    Parameters
    ----------
    
    Returns
    ----------

    """
    Mpl_GeV = 1.220910e19
    crit = S/(4 * np.log(phi0 * Mpl_GeV / xmin**2))

    return crit - 1


def findNucleationTemp(pot, Tmax: float, Tmin: float, ndim: int,
                       Ttol=1e-2, verbose=False) -> float:
    """description

    Parameters
    ----------

    Returns
    ----------

    """
    critdict = {}
    def crit(T, critdict=critdict, verbose=False):
        try:
            T = T[0]
        except:
            pass
        S, phi0 = getActionAtT(T, pot, ndim)
        if ndim == 3:
            S = S/T
        c = nuclCriterion(S, phi0, T, pot.xmin)
        critdict[T] = (S, phi0, c)
        if verbose:
            print(f"T = {T:2.5g}, S = {S:2.5g}, phi0 = {phi0:2.5g}, criterion = {c:2.5g}")
        return c

    def abort_fmin(T, critdict=critdict, verbose=False):
        T = T[0]
        S, phi0, _ = critdict[T]
        if nuclCriterion(S, phi0, T, pot.xmin) < 0:
            raise StopIteration(T)
        elif T < 0:
            raise StopIteration(T)

    # First find the minimal value of the tunneling criterion!
    try:
        res = optimize.minimize(crit, 0.1*(Tmin+Tmax), method='Nelder-Mead', bounds=[(Tmin, Tmax)],
                                tol=.1, callback=abort_fmin, args=(critdict, verbose))
        Tmin = res.x[0]
    except StopIteration as e:
        Tmin = e.args[0]



    if nuclCriterion(critdict[Tmin][0], critdict[Tmin][1], Tmin, pot.xmin) > 0:
        print("No tunneling possible, nucleation criterion not fulfulled!")
        return -1

    # import pdb
    # pdb.set_trace()

    # search for the smallest possible value of tmax:
    Tmax_min = Tmax
    for TT in critdict.keys():
        if critdict[TT][2] > 0 and TT < Tmax_min:
            Tmax_min = TT
    
    
    if verbose: 
        print(f"Criterion at Tmin = {Tmin:2.5g}: ", nuclCriterion(critdict[Tmin][0], critdict[Tmin][1], Tmin, pot.xmin))
        print(f"Criterion at Tmax = {Tmax_min:2.5g}: ", crit(Tmax_min))
    Tnuc = optimize.brentq(crit, Tmin, Tmax_min)
    if verbose:
        print("Criterion at Tnuc = ", nuclCriterion(critdict[Tnuc][0], critdict[Tnuc][1], Tnuc, pot.xmin))
    return Tnuc


def findNucleationTempTriangleApprox(pot, Tmax: float, Tmin: float, ndim: int,
                                     Ttol=1e-5, verbose=False) -> float:
    """description

    Parameters
    ----------

    Returns
    ----------

    """
    critdict = {}
    def crit(T, critdict=critdict, verbose=False):
        try:
            T = T[0]
        except:
            pass
        S, phi0 = triangleApproxAction(pot, T, ndim)
        if ndim == 3:
            S = S/T
        critdict[T] = (S, phi0)
        c = nuclCriterion(S, phi0, T, pot.xmin)
        if verbose:
            print(f"T = {T:2.5g}, S = {S:2.5g}, phi0 = {phi0:2.5g}, criterion = {c:2.5g}")
        return c

    def abort_fmin(T, critdict=critdict, verbose=False):
        T = T[0]
        S, phi0 = critdict[T]
        if nuclCriterion(S, phi0, T, pot.xmin) < 0:
            raise StopIteration(T)
        elif T < 0:
            raise StopIteration(T)

    # First find the minimal value of the tunneling criterion!
    try:
        res = optimize.minimize(crit, 0.1*(Tmin+Tmax), method='Nelder-Mead', bounds=[(Tmin, Tmax)],
                                tol=1e-8, callback=abort_fmin, args=(critdict, verbose))
        Tmin = res.x[0]
    except StopIteration as e:
        Tmin = e.args[0]



    if nuclCriterion(critdict[Tmin][0], critdict[Tmin][1], Tmin, pot.xmin) > 0:
        if verbose:
            print("No tunneling possible, nucleation criterion not fulfulled!")
        return -1

    if verbose: 
        print("Criterion at Tmin = ", nuclCriterion(critdict[Tmin][0], critdict[Tmin][1], Tmin, pot.xmin))
        print("Criterion at Tmax = ", crit(Tmax))
    Tnuc = optimize.brentq(crit, Tmin, Tmax)
    if verbose:
        print("Criterion at Tnuc = ", nuclCriterion(critdict[Tnuc][0], critdict[Tnuc][1], Tnuc, pot.xmin))
    return Tnuc


if __name__=="__main__":
    xmin = 2.5e3
    vir = 0.85
    eps = 0.05
    deltat = -0.5
    N = 4.5
    withQCD=True
    n = 0.3
    ndim = 4

    # test values
    ndim = 3
    deltat = -0.3
    n = 0.20
    eps = 0.01
    vir = 0.68965517

    delta = -.5 * vir**2
    
    pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=withQCD)
    Tcrit = np.power(-8* pot.VGW(xmin)/(np.pi**2 * N**2), 1/4.0)
    Tmax = (1-1e-3)*Tcrit
    Tnuc = findNucleationTemp(pot, Tmax, 1e-20, ndim, Ttol=1e-2, verbose=True)
    print(f"Tnuc numerical: {Tnuc:2.5g}")

    TnucApprox = findNucleationTempTriangleApprox(pot, Tmax, 1e-20, ndim, Ttol=1e-2, verbose=False)
    print(f"Tnuc approxima: {TnucApprox:2.5g}")
    
