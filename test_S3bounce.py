"""Module description

Author: Jonas Matuszak <jonas.matuszak@kit.edu>, 2025
"""

from calcBounce import *
from potential import Potential
import numpy as np

import matplotlib.pyplot as plt
import multiprocessing

from scipy import optimize


def testInstanton(T, xmin, vir, eps, delta, n, N, withQCD):
    pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=withQCD)
    pot.xmeta_min = -T

    

    V = lambda x: pot.Vfull(x, T)
    dV = lambda x: pot.dVfull(x, T)
    d2V = lambda x: pot.d2Vfull(x, T)
    phibar = optimize.minimize_scalar(lambda phi: -V(phi), bounds=(0, xmin)).x
    print("phibar = ", phibar)
    
    insta = Instanton(V, dV, d2V, ndim=3, Nkin=3*N**2/(2*np.pi**2))
    
    r, y, c = insta.findProfile(-T, xmin, phibar)
    phi, dphi = y.T
    S = insta.calcAction(r, phi, dphi, -T)
    
    # r, y, ctype = findProfile(V, dV, d2V, xmin,
    #                           pot.xmeta_min, N=4.5, rmin=1e-8, rmax=1e5,
    #                           ndim=3)
    # phi, dphi = y.T
    # phi0 = phi[0]
    # S = findActionRadion(r, phi, dphi, N, V, pot.xmeta_min, ndim=3)

    print("Action S3 = ", S)
    print("ctype = ", c)
    # print("New action old integration:", findActionRadion(rI, phiI, dphiI, N, V, pot.xmeta_min, ndim=3))
    # print("Old action S = ", S)
    # print("Old action new integration: = ", insta.calcAction(r, phi, dphi, -T))

    plt.semilogx(r, phi, label="New")
    # plt.semilogx(r, phi, label="Old")
    plt.show()

    insta4 = Instanton(V, dV, d2V, ndim=4, Nkin=3*N**2/(2*np.pi**2))

    r, y, c = insta.findProfile(-T, xmin, phibar)
    phi, dphi = y.T
    S4 = insta.calcAction(r, phi, dphi, -T)
    print("Action S3 = ", S)
    print("ctype = ", c)
    # print("New action old integration:", findActionRadion(rI, phiI, dphiI, N, V, pot.xmeta_min, ndim=3))
    # print("Old action S = ", S)
    # print("Old action new integration: = ", insta.calcAction(r, phi, dphi, -T))

    plt.semilogx(r, phi, label="New")
    # plt.semilogx(r, phi, label="Old")
    plt.show()


def testInstantonS4(T, xmin, vir, eps, delta, n, N, withQCD):
    pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=withQCD)

    V = lambda x: pot.Vfull(x, T)
    dV = lambda x: pot.dVfull(x, T)
    d2V = lambda x: pot.d2Vfull(x, T)
    phibar = optimize.minimize_scalar(lambda phi: -V(phi), bounds=(0, xmin),
                                      options={"xatol": 1e-15}).x
    print("phibar = ", phibar)
    phimeta = optimize.minimize_scalar(V, bounds=(0, phibar), options={"xatol": 1e-20}).x
    print("meta minimum = ", phimeta)

    # xx = np.logspace(np.log10(phimeta/2), np.log10(phimeta*2))
    # plt.semilogx(xx, V(xx))
    # plt.scatter(phimeta, V(phimeta))
    # plt.show()
    
    insta = Instanton(V, dV, d2V, ndim=4, Nkin=3*N**2/(2*np.pi**2), rmin_scaling=1e-8,
                      rmax_scaling=1e10, xtol=1e-20, phitol=1e-16)
    
    r, y, c = insta.findProfile(phimeta, xmin, phibar, f=1e-3, useInitialConditions=True)
    phi, dphi = y.T
    S = insta.calcAction(r, phi, dphi, phimeta)
    Salt = findActionRadion(r, phi, dphi, N, V, phimeta, ndim=4)
    

    print("Action S4 = ", S)
    print("Action S4 alt = ", Salt)
    print("ctype = ", c)

    # plt.semilogx(r, phi, label="New")
    # plt.show()
    return S, phi[0]


def getActionAtT(T, xmin, vir, eps, delta, n, N, withQCD):
    pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=withQCD)
    pot.xmeta_min = -T

    V = lambda x: pot.Vfull(x, T)
    dV = lambda x: pot.dVfull(x, T)
    d2V = lambda x: pot.d2Vfull(x, T)
    phibar = optimize.minimize_scalar(lambda phi: -V(phi), bounds=(0, xmin)).x

    insta = Instanton(V, dV, d2V, ndim=3, Nkin=3*N**2/(2*np.pi**2))
    r, y, ctype = insta.findProfile(-T, xmin, phibar)
    phi, dphi = y.T
    phi0 = phi[0]
    S = insta.calcAction(r, phi, dphi, -T)
    return S, phi0


def nucleationCriterionS3(T, xmin, vir, eps, delta, n, N, withQCD):
    """Nucleation criterion

    Parameters
    ----------

    Returns
    ----------

    """

    S3, phi0 = getActionAtT(T, xmin, vir, eps, delta, n, N, withQCD)
    S = S3/T

    Mpl_GeV = 1.220910e19

    crit = S/(4 * np.log(phi0 * Mpl_GeV / xmin**2))

    print(f"Nucleation critrion at T = {T:2.5g}: S3/T = {S:2.5g}, crit-1 = {crit-1:2.5g}.")
    
    return crit - 1


def getTnuc(xmin, vir, eps, delta, n, N, withQCD=True):
    pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=withQCD)
    Tcrit = np.power(-8* pot.VGW(xmin)/(np.pi**2 * N**2), 1/4.0)
    try:
        Tnuc = optimize.brentq(lambda T: nucleationCriterionS3(T, xmin, vir, eps, delta, n, N, withQCD),
                           1e-2, Tcrit)
        return Tnuc
    except Exception as e:
        print("Did not find a nucleation temperature:")
        print(e)
        return 0
    




def scanS3_T(Trange, vir, eps, delta, withQCD=True,n_jobs=14):
    xmin = 2.5e3
    N = 4.5
    n = 0.3

    input_params = []
    for i, T in enumerate(Trange):
        input_params.append((T, xmin, vir, eps, delta, n, N, withQCD))

    pool = multiprocessing.Pool(processes=n_jobs)
    res = pool.starmap(getActionAtT, input_params)
    pool.close()

    Srange = np.zeros_like(Trange)
    phi0range = np.zeros_like(Trange)
    for i in range(len(Trange)):
        Srange[i], phi0range[i] = res[i]

    return Trange, Srange
    

def test_scans(recompute=False):
    Trange = np.logspace(-2, 2)

    epsr = [0.01, 0.08, 0.1, 0.1, 0.1]
    virr = [1, 1, 1, 1, 0.5]
    deltar = [0,0,0,-.3, -.3]
    cols = ["C0","C1", "C2", "C3", "C4"]
    SvalsQCD = []
    Svals = []
    Tvals = []

    if recompute:
        for i in range(len(epsr)):
            _, SrQCD = scanS3_T(Trange, virr[i], epsr[i], deltar[i], withQCD=True)
            _, Sr = scanS3_T(Trange, virr[i], epsr[i], deltar[i], withQCD=False)
            SvalsQCD.append(SrQCD)
            Svals.append(Sr)
            Tvals.append(Trange)
        np.save("S3BounceActions.npy", np.asarray([Tvals, SvalsQCD, Svals]))

    Tvals, SvalsQCD, Svals = np.load("S3BounceActions.npy")
    for i in range(len(epsr)):
        lbl = r"$\epsilon = "+f"{epsr[i]:3.2f}~" + r"v_{ir} = " + f"{virr[i]:2.1f} ~" + r"\delta = " + f"{deltar[i]}$"
        plt.loglog(Tvals[i], SvalsQCD[i]/Tvals[i], color=cols[i], label=lbl)
        plt.loglog(Tvals[i], Svals[i]/Tvals[i], color=cols[i], ls="--")

    plt.xlabel("T [GeV]", fontsize=15)
    plt.ylabel(r"$S_3/T$", fontsize=15)
    plt.legend()
    plt.show()


def findRScale(V, phi0, phimeta, ndim):
    """This estimates the time the field needs to roll down to the minimum
    by assuming a linear potential between phi0 and phimeta.

    Parameters
    ----------
    V : function
        Returns the potential at V(phi)
    phi0 : float
        The field release point
    phimeta : float
        The meta stable minimum of the potential
    ndim : int
        Number of dimensions.
    
    The analytic
    solution to the equation

    .. math::
     \frac{d^2\phi}{dr^2} + \frac{\nu}{r}\frac{d\phi}{dr} = b

    is

    .. math::
     \phi(r) = \phi_0 + \frac{b}{2(1+\nu)} r^2
    
    Get the time that takes the field to roll down
    a linear potential."""
    nu = ndim - 1
    R = np.sqrt((phimeta-phi0)**2/(V(phimeta) - V(phi0)) * 2*(1+nu))
    return R

def test_calc():
    
    T = 10
    xmin = 2.5e3
    vir = 1
    epsilon = 1/20
    deltat = -.5
    N = 4.5
    n = 0.3

    delta = deltat * vir**2

    pot = Potential(xmin, vir, epsilon, delta, n, N=N, withQCD=False)
    pot.xmeta_min = -T

    V = lambda x: pot.Vfull(x, T)
    dV = lambda x: pot.dVfull(x, T)
    d2V = lambda x: pot.d2Vfull(x, T)

    
    phibar = pot.findBarrier(T)
    # phiroot = findPhiMin(V, phibar, pot.xmin+T, xtol=1e-15)
    phiroot = optimize.brentq(lambda x: V(x) - V(pot.xmeta_min), phibar, xmin -1e-15)

    phi_min = phiroot

    xrange = np.linspace(-T, 1.1*np.maximum(phiroot, pot.xc), 50)
    plt.plot(xrange, V(xrange), label='Full')
    plt.plot(xrange, pot.VQCD(xrange) - pot.VQCD0, label="QCD")
    plt.plot(xrange, pot.VGW(xrange), label="GW")
    plt.plot(phibar, V(phibar), 'o')
    plt.plot(phi_min, V(phi_min), 'o')
    plt.axhline(0, color="gray", alpha=0.2)
    plt.legend()
    plt.show()

    phitol = 1e-8
    xtol = 1e-15
    rmin = 1e-7
    rmax = 1e4

    phi_max = xmin

    xmin = xtol*10
    xmax = np.inf
    xincrease = 5.0
    x = 1

    # phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
    phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)

    # rscale1 = phibar / np.sqrt(6 * np.abs(V(phibar)))
    # rscale1 = phibar / np.power(6 * np.abs(V(phibar)), 1/3.0)
    rscale1 = findRScale(V, phi0, pot.xmeta_min, ndim=3)
    # rscale1 = 0.1
    print("rscale1 = ", rscale1)
    rscale = rscale1

    # rscale = np.sqrt(3 * phi0/np.abs(dV(phi0)))
    print("rscale = ", rscale)
    # rmin = rscale * rmin
    rmin_scaling = rmin
    rmax_scaling = rmax
    rmin = rscale * rmin_scaling
    rmax = rscale * rmax_scaling
    dr0 = rmin
    drmin = dr0 * 0.01
    epsabs = abs(np.array([xmin, xmin/rscale])*phitol)

    epsfrac = np.array([1,1]) * phitol

    # This can be improved
    # phi0 = xmin/2
    print("here")
    r0, phi0, dphi0 = initialConditions(V, dV, d2V, xmin, rmin, phi0, f=1e-2)
    # phi0, dphi0 = exactSolution(rmin, phi0, dV(phi0), d2V(phi0), ndim=3)

    # r0 = rmin
    print(f"Initial conditions: r0 = {r0:2.5g}, phi0 = {phi0:2.5g}, dphi0 = {dphi0:2.5g}, rscale = {rscale:2.5g}")

    y0 = np.array([phi0, dphi0])
    r, y, ctype = integrateProfile(r0, y0, dr0, dV, N, pot.xmeta_min, epsfrac, epsabs, drmin, rmax, ndim=3)

    it = 0
    while True:
        it += 1
        if it > 5000:
            break
        print("iteration = ", it)
        # phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
        phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)

        rscale = findRScale(V, phi0, pot.xmeta_min, ndim=3)
        print("Vmeta-V0 = ", V(pot.xmeta_min) - V(phi0))
        print("Vmeta-V0 = ", V(pot.xmeta_min) - V(phi0))
        rmin = rscale * rmin_scaling
        rmax = rscale * rmax_scaling
        dr0 = rmin
        drmin = dr0 * 0.01
        epsabs = abs(np.array([xmin, xmin/rscale])*phitol)
        # rscale = np.sqrt(3 * phi0/np.abs(dV(phi0)))
        # print("rscale = ", rscale)
        # rmin = rscale * rmin
        # rmax = rscale * rmax
        # dr0 = rmin
        # drmin = dr0 * 0.01
        # epsabs = abs(np.array([xmin, xmin/rscale])*phitol)

        # try to guess new rscale for new phi0
        print("#"*50)
        print(f"Searching initial conditions for phi0 = {phi0:2.5g}, rmin = {rmin:2.5g}")
        r0, phi0, dphi0 = initialConditions(V, dV, d2V, xmin, rmin, phi0, f=1e-2)
        # phi0, dphi0 = exactSolution(rmin, phi0, dV(phi0), d2V(phi0), ndim=3)
        y0 = np.array([phi0, dphi0])
        if not np.isfinite(r0) or not np.isfinite(x):
            # Use the last finite values instead
            # (assuming there are such values)
            print("fail!!")
            assert r is not None, "Failed to retrieve initial "\
                "conditions on the first try."
            break
        # print(f"Initial conditions: r0 = {r0:2.5g}, phi0 = {phi0:2.5g}, dphi0 = {dphi0:2.5g}")
        print(f"Initial conditions: r0 = {r0:2.5g}, phi0 = {phi0:2.5g}, dphi0 = {dphi0:2.5g}, rscale = {rscale:2.5g}")
        r, y, ctype = integrateAndSaveProfile(r0, y0, dr0, dV, N, pot.xmeta_min,
                                       epsfrac, epsabs, drmin, rmax, ndim=3)
        print("ctype = ", ctype)
        print(f"x = {x:2.5g}")
        print(f"xmin = {xmin:2.5g}")
        print(f"xmax = {xmax:2.5g}")
        print(f"phi0 = {phi0:2.5g}")
        if ctype == "converged":
            print("Converged !!!")
            print(r)
            print(y)
            break
        elif ctype == "undershoot":  # x is too big!
            xmax = x
            x = .5*(xmin+xmax)
        elif ctype == "overflow" or ctype == "overshoot": # x is too small!
            xmin = x
            print(f"Setting xmin to {xmin:2.5g}")
            x = x*xincrease if xmax == np.inf else .5*(xmin+xmax)

        # Reached tolerance?
        if (xmax-xmin) < xtol:
            print("Reached tolerance")
            print("y[0] = ", y[0])
            print("phimetamin = ", pot.xmeta_min)
            break

    r, y, ctype = integrateAndSaveProfile(r0, y0, dr0, dV, N, pot.xmeta_min,
                                          epsfrac, epsabs, drmin, rmax, ndim=3)

    print(f"xmetamin = {pot.xmeta_min:2.5g}")
    plt.semilogx(r, y[:, 0])
    plt.axvline(rscale1, ls="-.", color="gray", label="Rscale 1")
    plt.axvline(rscale, color="black", label="Rscale")
    plt.legend()
    plt.show()

    S3 = findActionRadion(r, y[:,0], y[:,1], N, V, pot.xmeta_min, ndim=3)


    return r, y, S3, pot


# fig, ax = plt.subplots(4,1, sharex=True)
# ax[0].semilogx(r, y[:,1])
# ax[0].set_ylabel(r"$d\phi/dr$")
# ax[1].semilogx(r, y[:,0])
# ax[1].set_ylabel(r"$\phi$")
# ax[2].semilogx(r, pot.Vfull(y[:,0], T=1))
# ax[2].set_ylabel(r"$V(\phi(r))$")
# ax[3].semilogx(r, pot.dVfull(y[:,0], T=1))
# ax[3].set_ylabel(r"$dV(\phi(r))/d\phi$")

# plt.show()
