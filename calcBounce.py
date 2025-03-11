"""This module calculates the bounce action for the radion potential

Mostly stolen from CosmoTransition, otherwise:

Author: Jonas Matuszak <jonas.matuszak@kit.edu>, 2025
"""
import numpy as np
from scipy import optimize
from scipy import special
from scipy import integrate
import matplotlib.pyplot as plt


class Instanton():
    def __init__(self, V, dV, d2V, ndim, Nkin, rmin_scaling=1e-4, rmax_scaling=1e4,
                 phitol=1e-8, xtol=1e-12):
        """Instanton solution 

        Parameters
        ----------
        V : function

        dV : function

        d2V : function

        ndim : int
            Number of dimension in which to solve the bounce action (3 or 4)

        Returns
        ----------

        """
        self.V = V
        self.dV = dV
        self.d2V = d2V
        self.rmin_scaling = rmin_scaling
        self.rmax_scaling = rmax_scaling
        self.phitol = phitol
        self.xtol = xtol
        if ndim != 3 and ndim != 4:
            raise Exception("Wrong numer of dimension, try ndim = 3 or ndim = 4!")
        self.ndim = ndim
        self.Nkin = Nkin # Normalisation of the kinetic term in the action

    def findRScale(self, phimeta, phi0):
        r"""This estimates the time the field needs to roll down to the minimum
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

        nu = self.ndim - 1
        R = np.sqrt((phimeta-phi0)**2/(self.V(phimeta) - self.V(phi0)) *2*(1+nu))
        return R

    def exactSolution(self, r, phi0, dV, d2V):
        r""" Find `phi(r)` given `phi(r=0)`, assuming a quadratic potential.

        Parameters
        ----------
        r : float
            The radius at which the solution should be calculated.
        phi0 : float
            The field at `r=0`.
        dV, d2V : float
            The potential's first and second derivatives evaluated at `phi0`.
        ndim : int, default ndim = 3
            Dimensions, either 3 or 4

        Returns
        -------
        phi, dphi : float
            The field and its derivative evaluated at `r`.

        Notes
        -----

        If the potential at the point :math:`\phi_0` is a simple quadratic, the
        solution to the instanton equation of motion can be determined exactly.
        The non-singular solution to

        .. math::
        \frac{d^2\phi}{dr^2} + \frac{\alpha}{r}\frac{d\phi}{dr} =
        V'(\phi_0) + V''(\phi_0) (\phi-\phi_0)

        is

        .. math::
        \phi(r)-\phi_0 = \frac{V'}{V''}\left[
        \Gamma(\nu+1)\left(\frac{\beta r}{2}\right)^{-\nu} I_\nu(\beta r) - 1
        \right]

        where :math:`\nu = \frac{\alpha-1}{2}`, :math:`I_\nu` is the modified
        Bessel function, and :math:`\beta^2 = V''(\phi_0) > 0`. If instead
        :math:`-\beta^2 = V''(\phi_0) < 0`, the solution is the same but with
        :math:`I_\nu \rightarrow J_\nu`.
        """
        np.seterr(over = 'ignore')
        beta = np.sqrt(abs(d2V))
        beta_r = beta*r
        np.seterr(over = 'warn')
        nu = 0.5 * (self.ndim - 1) # nu =  for S3
        gamma = special.gamma  # Gamma function
        iv, jv = special.iv, special.jv  # (modified) Bessel function
        if beta_r < 1e-2:
            # Use a small-r approximation for the Bessel function.
            s = +1 if d2V > 0 else -1
            phi = 0.0
            dphi = 0.0
            for k in range(1,4):
                _ = (0.5*beta_r)**(2*k-2) * s**k / (gamma(k+1)*gamma(k+1+nu))
                phi += _
                dphi += _ * (2*k)
            phi *= 0.25 * gamma(nu+1) * r**2 * dV * s
            dphi *= 0.25 * gamma(nu+1) * r * dV * s
            phi += phi0
        elif d2V > 0:
            import warnings
            # If beta_r is very large, this will throw off overflow and divide
            # by zero errors in iv(). It will return np.inf though, which is
            # what we want. Just ignore the warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi = (gamma(nu+1)*(0.5*beta_r)**-nu * iv(nu, beta_r)-1) * dV/d2V
                dphi = -nu*((0.5*beta_r)**-nu / r) * iv(nu, beta_r)
                dphi += (0.5*beta_r)**-nu * 0.5*beta \
                    * (iv(nu-1, beta_r)+iv(nu+1, beta_r))
                dphi *= gamma(nu+1) * dV/d2V
                phi += phi0
        else:
            phi = (gamma(nu+1)*(0.5*beta_r)**-nu * jv(nu, beta_r) - 1) * dV/d2V
            dphi = -nu*((0.5*beta_r)**-nu / r) * jv(nu, beta_r)
            dphi += (0.5*beta_r)**-nu * 0.5*beta \
                * (jv(nu-1, beta_r)-jv(nu+1, beta_r))
            dphi *= gamma(nu+1) * dV/d2V
            phi += phi0
        return phi, dphi

    def eqOfMotion(self, y, r):
        """Equations of motion

        Parameters
        ----------
        Returns
        ----------

        """
        phi = y[0]
        dphi = y[1]
        nu = self.ndim - 1
        return np.array([dphi, self.dV(phi)/self.Nkin - nu/r*dphi])

    def initialConditions(self, phi0, rmin, f=1e-2):
        r"""
        Finds the initial conditions for integration.

        The instanton equations of motion are singular at `r=0`, so we
        need to start the integration at some larger radius. This
        function finds the value `r0` such that `phi(r0) = phi_cutoff`.
        If there is no such value, it returns the intial conditions at `rmin`.

        Parameters
        ----------
        delta_phi0 : float
            `delta_phi0 = phi(r=0) - phi_absMin`
        rmin : float
            The smallest acceptable radius at which to start integration.
        delta_phi_cutoff : float
            The desired value for `phi(r0)`.
            `delta_phi_cutoff = phi(r0) - phi_absMin`.

        Returns
        -------
        r0, phi, dphi : float
            The initial radius and the field and its derivative at that radius.

        Notes
        -----
        The field values are calculated using :func:`exactSolution`.
        """
        dV0 = self.dV(phi0)
        d2V0 = self.d2V(phi0)

        phi_cutoff = phi0 * (1 - f)

        phi_r0, dphi_r0 = self.exactSolution(rmin, phi0, dV0, d2V0)
        if abs(phi_r0) < abs(phi_cutoff):
            # The initial conditions at rmin work. Stop here.
            return rmin, phi_r0, dphi_r0
        if np.sign(dphi_r0) > 0:
            # The field is evolving in the wrong direction.
            # Increasing r0 won't increase |delta_phi_r0|/
            return rmin, phi_r0, dphi_r0

        # Find the smallest r0 such that delta_phi_r0 > delta_phi_cutoff
        # print("rmin input: ", rmin)
        r = rmin
        while np.isfinite(r):
            rlast = r
            np.seterr(over='ignore')
            r *= 10
            np.seterr(over='warn')
            phi, dphi = self.exactSolution(r, phi0, dV0, d2V0)
            # print(f"r = {r:2.5g}, phi = {phi:2.5g}")
            if abs(phi) < abs(phi_cutoff):
                break

        # Now find where phi - self.phi_absMin = delta_phi_cutoff exactly

        def deltaPhiDiff(r_):
            p = self.exactSolution(r_, phi0, dV0, d2V0)[0]
            res = abs(p) - abs(phi_cutoff)
            if np.isnan(res):
                res = np.inf
            return res

        if deltaPhiDiff(rlast) == np.inf:
            return rmin, phi0, 0.
        
        r0 = optimize.brentq(deltaPhiDiff, rlast, r, disp=False)
        phi_r0, dphi_r0 = self.exactSolution(r0, phi0, dV0, d2V0)
        return r0, phi_r0, dphi_r0

    def integrateProfile(self, r0, y0, dr0, drmin, rmax, phimeta, epsfrac, epsabs):
        """description

        Parameters
        ----------

        Returns
        ----------
        """
        dydr0 = self.eqOfMotion(y0, r0)
        ysign = np.sign(y0[0] - phimeta)
        dr = dr0

        convergence_type = None
        while True:
            dy, dr, drnext = rkqs(y0, dydr0, r0, self.eqOfMotion,
                                  dr, epsfrac, epsabs)
            r1 = r0 + dr
            y1 = y0 + dy

            dydr1 = self.eqOfMotion(y1, r1)
            if (r1 > rmax):
                if r1 > 1e100:
                    raise Exception("Integration: r > 1e100. Something is going very wrong!")
                rmax = rmax**2
                # raise Exception("Integration: r > rmax")
            elif (dr < drmin):
                print("Integration: dr < drmin")
                raise Exception(f"Integration: substepping min stepsizde drmin = {drmin:2.5g}")
            # are the conditions at r -> infty fulfilled?
            elif (abs(y1 - np.array([phimeta, 0])) < 3*epsabs).all():
                r, y = r1, y1
                convergence_type = "converged"
                break
            elif y1[1] * ysign > 0 or (y1[0]-phimeta) * ysign < 0:
                f = cubicInterpFunction(y0, dr*dydr0, y1, dr*dydr1)
                if (y1[1] * ysign > 0):
                    # Extrapolate to where dphi(r) = 0
                    try:
                        x = optimize.brentq(lambda x: f(x)[1], 0, 1)
                    except:
                        x = r1
                    convergence_type = "undershoot"
                else:
                    # Extrapolate to where phi(r) = phi_metaMin
                    try:
                        x = optimize.brentq(lambda x: f(x)[0]-phimeta, 0,1)
                    except:
                        x = r1
                    convergence_type = "overshoot"
                r = r0 + dr*x
                y = f(x)
                break
            # Advance the integration variables
            r0, y0, dydr0 = r1, y1, dydr1
            dr = drnext

        # Check convergence for a second time.
        # The extrapolation in overshoot/undershoot might have gotten us within
        # the acceptable error.
        if (abs(y - np.array([phimeta, 0])) < 3 * epsabs).all():
            convergence_type = "converged"
        return r0, y0, convergence_type


    def integrateAndSaveProfile(self, r0, y0, dr0, drmin, rmax, phimeta, epsfrac, epsabs):
        """Integrate the profile and save to an array

        TODO: change to use a fixed evaluation array in r

        Parameters
        ----------

        Returns
        ----------

        """
        dydr0 = self.eqOfMotion(y0, r0)
        ysign = np.sign(y0[0] - phimeta)
        dr = dr0

        yrange = [y0]
        rrange = [r0]

        convergence_type = None
        while True:
            dy, dr, drnext = rkqs(y0, dydr0, r0, self.eqOfMotion,
                                  dr, epsfrac, epsabs)
            r1 = r0 + dr
            y1 = y0 + dy
        
            dydr1 = self.eqOfMotion(y1, r1)
            if (r1 > rmax):
                if r1 > 1e100:
                    raise Exception("Integration: r > 1e100. Something is going very wrong!")
                rmax = rmax**2
                # raise Exception("Integration: r > rmax")
            elif (dr < drmin):
                print("Integration: dr < drmin")
                raise Exception(f"Integration: substepping min stepsizde drmin = {drmin:2.5g}")
            # are the conditions at r -> infty fulfilled?
            elif (abs(y1 - np.array([phimeta, 0])) < 3*epsabs).all():
                r, y = r1, y1
                convergence_type = "converged"
                break
            elif y1[1] * ysign > 0 or (y1[0]-phimeta) * ysign < 0:
                f = cubicInterpFunction(y0, dr*dydr0, y1, dr*dydr1)
                if (y1[1] * ysign > 0):
                    # Extrapolate to where dphi(r) = 0
                    try:
                        x = optimize.brentq(lambda x: f(x)[1], 0, 1)
                    except:
                        x = r1
                    convergence_type = "undershoot"
                else:
                    # Extrapolate to where phi(r) = phi_metaMin
                    try:
                        x = optimize.brentq(lambda x: f(x)[0]-phimeta, 0,1)
                    except:
                        x = r1
                    convergence_type = "overshoot"
                r = r0 + dr*x
                y = f(x)
                break
            # Advance the integration variables
            r0, y0, dydr0 = r1, y1, dydr1
            dr = drnext
            rrange.append(r1)
            yrange.append(y1)

        # Check convergence for a second time.
        # The extrapolation in overshoot/undershoot might have gotten us within
        # the acceptable error.
        if (abs(y - np.array([phimeta, 0])) < 3 * epsabs).all():
            convergence_type = "converged"
        return np.asarray(rrange), np.asarray(yrange), convergence_type

    def findProfile(self, phimeta, vev, phibar, f=1e-3, useInitialConditions=True):
        """Find the instanton profile minimizing the action

        Parameters
        ----------
        phimeta : float
            Field value of the metastable minimum
        vev : float
            Field value of the global minimum
        phibar : float
            Field value of the potential barrier

        Returns
        ----------

        """
        phiroot = optimize.brentq(lambda phi: self.V(phi) - self.V(phimeta), phibar, vev - 1e-15)
        phi_min = phiroot
        phi_max = vev

        xmin = self.xtol*10
        xmax = np.inf
        xincrease = 5.0
        x = 1

        epsfrac = np.array([1, 1]) * self.phitol

        it = 0
        while True:
            it += 1
            if it > 500:
                break
            phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
            rscale = self.findRScale(phimeta, phi0)
            rmin = rscale * self.rmin_scaling
            rmax = rscale * self.rmax_scaling
            dr0 = rmin
            drmin = dr0 * 1e-4
            epsabs = abs(np.array([vev, vev/rscale])*self.phitol)
            if useInitialConditions:
                r0, phi0, dphi0 = self.initialConditions(phi0, rmin, f=f)
            else:
                r0, phi0, dphi0 = rmin, phi0, 0.0
            y0 = np.array([phi0, dphi0])
            r, y, ctype = self.integrateProfile(r0, y0, dr0, drmin, rmax, phimeta,
                                                epsfrac, epsabs)
            if ctype == "converged":
                break
            elif ctype == "undershoot":  # x is too big!
                xmax = x
                x = .5*(xmin+xmax)
            elif ctype == "overflow" or ctype == "overshoot": # x is too small!
                xmin = x
                x = x*xincrease if xmax == np.inf else .5*(xmin+xmax)

            # Reached tolerance?
            if (xmax-xmin) < self.xtol:
                # print("Reached tolerance")
                break

        r, y, ctype = self.integrateAndSaveProfile(r0, y0, dr0, drmin, rmax, phimeta,
                                                   epsfrac, epsabs)

        # plt.semilogx(r, y[:,0])
        # plt.axvline(rscale)
        # plt.show()
        
        return r, y, ctype

    def calcAction(self, r, phi, dphi, phimeta):
        """Calculate the action from a bubble profile.

        Parameters
        ----------
        r : ndarray
        phi : ndarray
        dphi : ndarray
        phimeta : float
            Field value of the meta stable minimum

        Returns
        S : float
        """
        d = self.ndim
        area = r**(d-1) * 2*np.pi**(d*.5)/special.gamma(d*.5)
        integrand = self.Nkin * 0.5 * dphi**2 + (self.V(phi) - self.V(phimeta))
        integrand *= area
        S = integrate.simpson(integrand, x=r)
        volume = r[0]**d * np.pi**(d*.5)/special.gamma(d*.5 + 1)
        S += volume * (self.V(phi[0]) - self.V(phimeta))
        return S



def S4Approx(pot, T):

    V = lambda x: pot.Vfull(x, T)
    phibar = optimize.minimize_scalar(lambda phi: -V(phi), bounds=(0, pot.xmin),
                                      options={"xatol": 1e-20}).x

    def S4apr(x):
        S = 9 * pot.N**4 / (8*np.pi) * x**4/(- V(x))
        return S
    res = optimize.minimize_scalar(S4apr, bounds=(phibar, pot.xmin))
    S4 = res.fun
    phi0 = res.x
    return S4, phi0


def S3Approx(pot, T):

    V = lambda x: pot.Vfull(x, T)
    Tcrit = np.power(-8 * pot.VGW(pot.xmin)/(np.pi**2 * pot.N**2), 1/4.0)
    phibar = optimize.minimize_scalar(lambda phi: -V(phi), bounds=(0, pot.xmin),
                                      options={"xatol": 1e-20}).x

    def S3apr(x):
        S = np.sqrt(3)/np.pi**2 * pot.N**3 * x**3
        S /= np.sqrt(V(pot.xmin) * (T/Tcrit)**4 - V(x))
        return S
    res = optimize.minimize_scalar(S3apr, bounds=(phibar, pot.xmin + T))
    S3 = res.fun
    phi0 = res.x
    return S3, phi0


def triangleApproxAction(pot, T: float, d: int, debug=False) -> (float, float):
    gamma = special.gamma
    phitol = 1e-8
    
    V = lambda x: pot.Vfull(x, T)
    phiT = optimize.minimize_scalar(lambda phi: -V(phi), bounds=(-T, pot.xmin),
                                      options={"xatol": 1e-20}).x

    phip = -T
    phim = pot.xmin
    lplus = (V(phiT) - V(phip))/(phiT- phip)

    phim = 2 * phiT
    # Approximate the slope with the release point as the reference point
    it = 0
    while True:
        it += 1
        lminus = - (V(phiT) - V(phim)) / (phiT - phim)
        c = lminus/lplus

        alpha = (2 * c - d*((c + 1)**(2/d) - 1))/(2*d*(d-2))

        # releasepoint 
        phi0 = phiT + c/(2*d*alpha) * (phiT - phip)

        S = 4 * (c + 1)/(d *(d + 2) *gamma(d/2)) \
            * (2*np.pi * (d - 2) * d / (2*c- d*((c+1)**(2/d) - 1)))**(d/2) \
            * (phiT - phip)**d / (V(phiT) - V(phip))**((d-2)/2)

        if np.abs(phi0 - phim) < phitol:
            break
        elif it > 1000:
            break

        phim = phi0 + (phim- phi0)*.5

        # phim = 2 * phiT
        # break

    if debug:
        phirange = np.linspace(phip, phi0*1.05, 200)
        phiprange = np.linspace(phip, phiT, 100)
        phimrange = np.linspace(phiT, phim, 100)
        plt.plot(phirange, V(phirange))
        plt.scatter(phiT, V(phiT), label=r"$\phi$ barrier")
        plt.scatter(phi0, V(phi0), label=r"$\phi_0$")
        plt.plot(phiprange, V(phip) + lplus*(phiprange - phip), color="green")
        plt.plot(phimrange, V(phiT)  + lminus*(phiT - phimrange), color="green")
        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$V(\phi)$")

        plt.legend()
        plt.show()

    return S, phi0

def testTriangle():
    from potential import Potential
    xmin = 2.5e3
    delta = -.5
    n = 0.3
    N = 4.5
    vir = 1
    eps = 1/20
    pot = Potential(xmin, vir, eps, delta, n)
    T = 1e-4

    S4, phi40 = triangleApproxAction(pot, T, 4, debug=True)
    print("O(4): phi0 = ", phi40, " S4 = ", S4)
    S3, phi30 = triangleApproxAction(pot, T, 3, debug=True)
    print("O(3): phi0 = ", phi30, " S3/T = ", S3/T)


def test_rscale(vir, epsilon, withQCD=True):
    # from radionPotential import RadionPotential
    
    from potential import Potential
    xmin = 2.5e3
    n = 0.6
    deltat = -0.5
    N = 4.5
    delta = vir**2 * deltat
    

    from radionPotential import RadionPotential
    pot = RadionPotential(xmin, vir, epsilon, delta, n, N=N, withQCD=withQCD)
    V = pot.Vfull
    dV = pot.dVfull
    d2V = pot.d2Vfull
    phibar = pot.xbar
    # pot = Potential(xmin, vir, epsilon, delta, n, N=N, withQCD=withQCD)
    # V = lambda x: pot.Vfull(x, 0)
    # dV = lambda x: pot.dVfull(x, 0)
    # d2V = lambda x: pot.d2Vfull(x, 0)
    # phibar = pot.xbar0

    phiroot = findPhiMin(V, phibar, xmin, xtol=1e-15)

    phi_min = phiroot

    xrange = np.linspace(0, 1.1*np.maximum(phiroot, pot.xc), 50)
    plt.plot(xrange, V(xrange), label='Full')
    plt.plot(xrange, pot.VQCD(xrange) - pot.VQCD0, label="QCD")
    plt.plot(xrange, pot.VGW(xrange), label="GW")
    plt.plot(phibar, V(phibar), 'o')
    plt.plot(phi_min, V(phi_min), 'o')
    plt.axhline(0, color="gray", alpha=0.2)
    plt.legend()
    plt.show()

    phitol=1e-11
    xtol = 1e-11
    rmin = 1e-10
    rmax = 1e4

    phi_max = xmin

    xmin = xtol*10
    xmax = np.inf
    xincrease = 5.0
    x = 1

    # phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
    phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)

    rscale1 = phibar / np.sqrt(6 * np.abs(V(phibar)))
    print("rscale1 = ", rscale1)
    rscale2 = findProfileScale_dV(phibar, d2V)
    print("rscale2 = ", rscale2)

    rscale = 10

    # rscale = np.sqrt(3 * phi0/np.abs(dV(phi0)))
    print("rscale = ", rscale)
    # rmin = rscale * rmin
    rmin = 1e-10
    rmax = rscale * rmax
    dr0 = rmin
    drmin = dr0 * 0.01
    epsabs = abs(np.array([xmin, xmin/rscale])*phitol)

    epsfrac = np.array([1,1]) * phitol

    # This can be improved
    # phi0 = xmin/2
    print("here")
    # r0, phi0, dphi0 = initialConditions(V, dV, d2V, xmin, rmin, phi0, f=1e-3)
    phi0, dphi0 = exactSolution(rmin, phi0, dV(phi0), d2V(phi0))

    r0 = rmin
    print(f"Initial conditions: r0 = {r0:2.5g}, phi0 = {phi0:2.5g}, dphi0 = {dphi0:2.5g}")

    y0 = np.array([phi0, dphi0])
    r, y, ctype = integrateAndSaveProfile(r0, y0, dr0, dV, N, pot.xmeta_min, epsfrac, epsabs, drmin, rmax)

    it = 0
    while True:
        it += 1
        if it > 5000:
            break
        print("iteration = ", it)
        # phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
        phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
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
        # r0, phi0, dphi0 = initialConditions(V, dV, d2V, xmin, rmin, phi0, f=1e-3)
        phi0, dphi0 = exactSolution(rmin, phi0, dV(phi0), d2V(phi0))
        y0 = np.array([phi0, dphi0])
        if not np.isfinite(r0) or not np.isfinite(x):
            # Use the last finite values instead
            # (assuming there are such values)
            print("fail!!")
            assert r is not None, "Failed to retrieve initial "\
                "conditions on the first try."
            break
        
        print(f"Initial conditions: r0 = {r0:2.5g}, phi0 = {phi0:2.5g}, dphi0 = {dphi0:2.5g}")
        r, y, ctype = integrateAndSaveProfile(r0, y0, dr0, dV, N, pot.xmeta_min,
                                              epsfrac, epsabs, drmin, rmax)
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
            print("y[-1,0] = ", y[-1,0])
            print("phimetamin = ", pot.xmeta_min)
            break

    y = np.asarray(y)
    r = np.asarray(r)

    print(f"xmetamin = {pot.xmeta_min:2.5g}")
    plt.semilogx(r, y[:, 0])
    # plt.axvline(rscale1, ls="-.", color="gray", label="Rscale 1")
    # plt.axvline(rscale2, ls="--", color="black", label="Rscale 2")
    plt.axvline(rscale, color="black", label="Rscale")
    plt.legend()
    plt.show()

    Sfull = findActionRadion(r, y[:,0], y[:,1], N, V, pot.xmeta_min)
    SGW = findActionRadion(r, y[:,0], y[:,1], N, pot.VGW, pot.xmeta_min)

    print(f"Action S4 = {Sfull:2.5g}")
    
    return r, y, Sfull, SGW, pot


def findRScale(V, phi0, phimeta, ndim):
    r"""This estimates the time the field needs to roll down to the minimum
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

def findProfile(V, dV, d2V, xmin, xmetamin, N, rmin=1e-5, rmax=1e5, ndim=4):
    """description

    Parameters
    ----------

    Returns
    ----------

    """
    phibar = findBarrier(V, xmin, xtol=1e-15)
    try:
        # phiroot = findPhiMin(V, phibar, xmin, xtol=1e-15)
        phiroot = optimize.brentq(lambda x: V(x) - V(xmetamin), phibar, xmin -1e-15)
    except:
        phiroot = phibar
    phi_min = phiroot #  - phibar

    phitol=1e-8
    xtol = 1e-12

    # phi_min = phibar
    phi_max = xmin

    xmin = xtol*10
    xmax = np.inf
    xincrease = 5.0
    x = 1

    # phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
    phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)

    # rscale = 1
    rmin_scaling = rmin
    rmax_scaling = rmax
    rscale = findRScale(V, phi0, xmetamin, ndim=ndim)
    # rscale = phibar / np.power(6 * np.abs(V(phibar)), 1/3.0)
    rmin = rscale * rmin_scaling
    
    rmax = rscale * rmax_scaling
    dr0 = rmin
    drmin = dr0 * 0.01
    epsabs = abs(np.array([xmin, xmin/rscale])*phitol)

    epsfrac = np.array([1,1]) * phitol

    r0, phi0, dphi0 = initialConditions(V, dV, d2V, xmin, rmin, phi0, f=1e-3)
    # phi0, dphi0 = exactSolution(rmin, phi0, dV(phi0), d2V(phi0), ndim=ndim)
    # r0 = rmin
    y0 = np.array([phi0, dphi0])
    r, y, ctype = integrateProfile(r0, y0, dr0, dV, N, xmetamin,
                                   epsfrac, epsabs, drmin, rmax, ndim=ndim)

    it = 0
    while True:
        it += 1
        if it > 500:
            break
        phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
        rscale = findRScale(V, phi0, xmetamin, ndim=ndim)
        rmin = rscale * rmin_scaling
        rmax = rscale * rmax_scaling
        dr0 = rmin
        drmin = dr0 * 0.01
        epsabs = abs(np.array([xmin, xmin/rscale])*phitol)

        r0, phi0, dphi0 = initialConditions(V, dV, d2V, xmin, rmin, phi0, f=1e-4)
        # phi0, dphi0 = exactSolution(rmin, phi0, dV(phi0), d2V(phi0), ndim=ndim)
        y0 = np.array([phi0, dphi0])
        if not np.isfinite(r0) or not np.isfinite(x):
            # Use the last finite values instead
            # (assuming there are such values)
            print("fail!!")
            assert r is not None, "Failed to retrieve initial "\
                "conditions on the first try."
            break

        r, y, ctype = integrateProfile(r0, y0, dr0, dV, N, xmetamin, epsfrac,
                                       epsabs, drmin, rmax, ndim=ndim)
        if ctype == "converged":
            print("Converged!")
            break
        elif ctype == "undershoot":  # x is too big!
            xmax = x
            x = .5*(xmin+xmax)
        elif ctype == "overflow" or ctype == "overshoot": # x is too small!
            xmin = x
            x = x*xincrease if xmax == np.inf else .5*(xmin+xmax)

        # Reached tolerance?
        if (xmax-xmin) < xtol:
            print("Reached tolerance")
            break

    r, y, ctype = integrateAndSaveProfile(r0, y0, dr0, dV, N, xmetamin, epsfrac,
                                          epsabs, drmin, rmax, ndim=ndim)
    return r, y, ctype


def findBarrier(V, xmax, xtol=1e-10):
    """Find the location of the potential barrier

    Parameters
    ----------
    V : function
        Potential(x, T)

    Returns
    ----------
    xbar : float
    """
    # xleft = xmin
    # xright = xmax
    # Vleft = V(xleft)
    # Vright = V(xright)
    # # binary search to find the maximum
    # while (abs(xleft - xright) > xtol):
    #     if (Vleft > Vright):
    #         xright = (xleft + xright) / 2.0
    #         Vright = V(xright)
    #     else:
    #         xleft = (xleft + xright) / 2.0
    #         Vleft = V(xleft)
    # xbar = (xleft + xright) / 2.0
    # if (abs(xbar - xmin) < xtol):
    #     w = Warning("Did not find a potential barrier!")
    #     print(w)
    #     return None
    # if (abs(xbar - xmax) < xtol):
    #     w = Warning("Did not find a potential barrier!")
    #     print(w)
    #     return None
    res = optimize.minimize_scalar(lambda x: -V(x), bounds=(0, xmax))
    if not res.success:
        raise Exception("Did not find potential barrier!")
    xbar = res.x
    return xbar


def findPhiMin(V, xbar, vev, xtol=1e-10):
    """Find the location of the smallest release
    point. 

    Parameters
    ----------
    V : function
        Potential(x, T)

    Returns
    ----------
    xbar : float
    """

    phiMin = optimize.brentq(V, xbar, vev -1e-15)

    return phiMin


def findProfileScale(V, vev, xtol=1e-15):
    """Find the charactericstic scale of the bubble radius"""

    xbar = findBarrier(V, vev, xtol=xtol)
    # Rscale = xbar / np.sqrt(6 * np.abs(V(xbar)))
    Rscale = xbar / np.power(6 * np.abs(V(xbar)), 1/3.0)
    return Rscale


def findProfileScale_dV(phibar, ddV, xtol=1e-10):
    """Find the charactericstic scale of the bubble radius"""
    Rscale = 2*np.pi /np.sqrt(np.abs(ddV(phibar)))
    return Rscale



def EOM_S4(y, r, dV, N):
    """Equations of motion for the O(4) symmetric bounce action"""
    phi = y[0]
    dphi = y[1]
    return np.array([dphi, 2*np.pi**2/(3*N**2)*dV(phi) - 3/r*dphi])


def EOM_S3(y, r, dV, N):
    """Equations of motion for the O(3) symmetric bounce action"""
    phi = y[0]
    dphi = y[1]
    return np.array([dphi, 2*np.pi**2/(3*N**2)*dV(phi) - 2/r*dphi])


def integrateProfile(r0, y0, dr0, dV, N, phi_metamin, epsfrac, epsabs, drmin, rmax, ndim=4):
    """"""
    if ndim == 4:
        def dY(y, r):
            return EOM_S4(y, r, dV, N)
    elif ndim == 3:
        def dY(y, r):
            return EOM_S3(y, r, dV, N)
    else:
        raise Exception("Wrong number of dimensions, either 3 or 4.")

    dydr0 = dY(y0, r0)
    ysign = np.sign(y0[0] - phi_metamin)
    dr = dr0

    convergence_type = None
    while True:
        dy, dr, drnext = rkqs(y0, dydr0, r0, dY, dr, epsfrac, epsabs)
        r1 = r0 + dr
        y1 = y0 + dy
        
        dydr1 = dY(y1, r1)
        if (r1 > rmax):
            if r1 > 1e100:
                raise Exception("Integration: r > 1e100. Something is going very wrong!")
            rmax = rmax**2
            # raise Exception("Integration: r > rmax")
        elif (dr < drmin):
            print("Integration: dr < drmin")
            raise Exception(f"Integration: substepping min stepsizde drmin = {drmin:2.5g}")
        # are the conditions at r -> infty fulfilled?
        elif (abs(y1 - np.array([phi_metamin, 0])) < 3*epsabs).all():
            r, y = r1, y1
            convergence_type = "converged"
            break
        elif y1[1] * ysign > 0 or (y1[0]-phi_metamin) * ysign < 0:
            f = cubicInterpFunction(y0, dr*dydr0, y1, dr*dydr1)
            if (y1[1] * ysign > 0):
                # Extrapolate to where dphi(r) = 0
                try:
                    x = optimize.brentq(lambda x: f(x)[1], 0, 1)
                except:
                    x = r1
                convergence_type = "undershoot"
            else:
                # Extrapolate to where phi(r) = phi_metaMin
                try:
                    x = optimize.brentq(lambda x: f(x)[0]-phi_metamin, 0,1)
                except:
                    x = r1
                convergence_type = "overshoot"
            r = r0 + dr*x
            y = f(x)
            break
        # Advance the integration variables
        r0, y0, dydr0 = r1, y1, dydr1
        dr = drnext

    # Check convergence for a second time.
    # The extrapolation in overshoot/undershoot might have gotten us within
    # the acceptable error.
    if (abs(y - np.array([phi_metamin, 0])) < 3 * epsabs).all():
        convergence_type = "converged"
    return r0, y0, convergence_type


def integrateAndSaveProfile(r0, y0, dr0, dV, N, phi_metamin, epsfrac, epsabs, drmin, rmax, ndim=4):
    """"""
    if ndim == 4:
        def dY(y, r):
            return EOM_S4(y, r, dV, N)
    elif ndim == 3:
        def dY(y, r):
            return EOM_S3(y, r, dV, N)
    else:
        raise Exception("Wrong number of dimensions, either 3 or 4.")

    dydr0 = dY(y0, r0)
    ysign = np.sign(y0[0] - phi_metamin)
    dr = dr0

    yrange = [y0]
    rrange = [r0]

    convergence_type = None
    while True:
        dy, dr, drnext = rkqs(y0, dydr0, r0, dY, dr, epsfrac, epsabs)
        r1 = r0 + dr
        y1 = y0 + dy

        dydr1 = dY(y1, r1)
        if (r1 > rmax):
            if r1 > 1e100:
                plt.semilogx(rrange, np.asarray(yrange)[:,0])
                plt.show()
                raise Exception("Integration: r > 1e100. Something is going very wrong!")
            rmax = rmax**2
        elif (dr < drmin):
            print("Integration: dr < drmin")
            return rrange, yrange, "substepping min stepsize"
        # are the conditions at r -> infty fulfilled?
        elif (abs(y1 - np.array([phi_metamin, 0])) < 3*epsabs).all():
            r, y = r1, y1
            convergence_type = "converged"
            break
        elif y1[1] * ysign > 0 or (y1[0]-phi_metamin) * ysign < 0:
            f = cubicInterpFunction(y0, dr*dydr0, y1, dr*dydr1)
            if (y1[1] * ysign > 0):
                # Extrapolate to where dphi(r) = 0
                try:
                    x = optimize.brentq(lambda x: f(x)[1], 0, 1)
                except:
                    x = r1
                convergence_type = "undershoot"
            else:
                # Extrapolate to where phi(r) = phi_metaMin
                try:
                    x = optimize.brentq(lambda x: f(x)[0]-phi_metamin, 0,1)
                except:
                    x = r1
                convergence_type = "overshoot"
            r = r0 + dr*x
            y = f(x)
            break
        # Advance the integration variables
        r0, y0, dydr0 = r1, y1, dydr1
        dr = drnext
        rrange.append(r1)
        yrange.append(y1)

    # Check convergence for a second time.
    # The extrapolation in overshoot/undershoot might have gotten us within
    # the acceptable error.
    if (abs(y - np.array([phi_metamin, 0])) < 3 * epsabs).all():
        convergence_type = "converged"
    return np.asarray(rrange), np.asarray(yrange), convergence_type



def findActionRadion(r, phi, dphi, N, V, phi_metamin, ndim=4):
    """Calculate the action from a bubble profile.

    This is for the Radion potential only.
    
    """
    d = ndim
    # area = 2 * np.pi**2 * r**(d-1)
    area = r**(d-1) * 2*np.pi**(d*.5)/special.gamma(d*.5)
    integrand = 3*N**2/(2*np.pi**2) * 0.5 * dphi**2 + (V(phi) - V(phi_metamin))
    integrand *= area
    S = integrate.simpson(integrand, x=r)
    volume = r[0]**d * np.pi**(d*.5)/special.gamma(d*.5 + 1)
    S += volume * (V(phi[0]) - V(phi_metamin))
    return S
    

def initialConditions(V, dV, d2V, vev, rmin, phi0, f=1e-2):
    r"""
    Finds the initial conditions for integration.

    The instanton equations of motion are singular at `r=0`, so we
    need to start the integration at some larger radius. This
    function finds the value `r0` such that `phi(r0) = phi_cutoff`.
    If there is no such value, it returns the intial conditions at `rmin`.

    Parameters
    ----------
    delta_phi0 : float
        `delta_phi0 = phi(r=0) - phi_absMin`
    rmin : float
        The smallest acceptable radius at which to start integration.
    delta_phi_cutoff : float
        The desired value for `phi(r0)`.
        `delta_phi_cutoff = phi(r0) - phi_absMin`.

    Returns
    -------
    r0, phi, dphi : float
        The initial radius and the field and its derivative at that radius.

    Notes
    -----
    The field values are calculated using :func:`exactSolution`.
    """
    # Start from phi = vev
    dV0 = dV(phi0)
    d2V0 = d2V(phi0)

    phi_cutoff = phi0 * (1 - f)

    phi_r0, dphi_r0 = exactSolution(rmin, phi0, dV0, d2V0)
    if abs(phi_r0) < abs(phi_cutoff):
        # The initial conditions at rmin work. Stop here.
        return rmin, phi_r0, dphi_r0
    if np.sign(dphi_r0) > 0:
        # The field is evolving in the wrong direction.
        # Increasing r0 won't increase |delta_phi_r0|/
        return rmin, phi_r0, dphi_r0

    # Find the smallest r0 such that delta_phi_r0 > delta_phi_cutoff
    # print("rmin input: ", rmin)
    r = rmin
    while np.isfinite(r):
        rlast = r
        np.seterr(over='ignore')
        r *= 10
        np.seterr(over='warn')
        phi, dphi = exactSolution(r, phi0, dV0, d2V0)
        # print(f"r = {r:2.5g}, phi = {phi:2.5g}")
        if abs(phi) < abs(phi_cutoff):
            break

    # Now find where phi - self.phi_absMin = delta_phi_cutoff exactly

    def deltaPhiDiff(r_):
        p = exactSolution(r_, phi0, dV0, d2V0)[0]
        res = abs(p) - abs(phi_cutoff)
        if np.isnan(res):
            res = np.inf
        return res

    # print("rlast = ", rlast)
    # print("dphidiff at rlast = ", deltaPhiDiff(rlast))
    # print("r = ", r)
    # print("dphidiff at r = ", deltaPhiDiff(r))
    # print("dphidiff at r_mid = ", deltaPhiDiff((r+rlast)/2))
    r0 = optimize.brentq(deltaPhiDiff, rlast, r, disp=False)
    # print(f"r0 = {r0:2.5g}")

    phi_r0, dphi_r0 = exactSolution(r0, phi0, dV0, d2V0)
    return r0, phi_r0, dphi_r0


def exactSolution(r, phi0, dV, d2V, ndim=3):
    R"""
    Find `phi(r)` given `phi(r=0)`, assuming a quadratic potential.

    Parameters
    ----------
    r : float
        The radius at which the solution should be calculated.
    phi0 : float
        The field at `r=0`.
    dV, d2V : float
        The potential's first and second derivatives evaluated at `phi0`.

    ndim : int, default ndim = 3
        Dimensions, either 3 or 4
    
    Returns
    -------
    phi, dphi : float
        The field and its derivative evaluated at `r`.

    Notes
    -----

    If the potential at the point :math:`\phi_0` is a simple quadratic, the
    solution to the instanton equation of motion can be determined exactly.
    The non-singular solution to

    .. math::
      \frac{d^2\phi}{dr^2} + \frac{\alpha}{r}\frac{d\phi}{dr} =
      V'(\phi_0) + V''(\phi_0) (\phi-\phi_0)

    is

    .. math::
      \phi(r)-\phi_0 = \frac{V'}{V''}\left[
      \Gamma(\nu+1)\left(\frac{\beta r}{2}\right)^{-\nu} I_\nu(\beta r) - 1
      \right]

    where :math:`\nu = \frac{\alpha-1}{2}`, :math:`I_\nu` is the modified
    Bessel function, and :math:`\beta^2 = V''(\phi_0) > 0`. If instead
    :math:`-\beta^2 = V''(\phi_0) < 0`, the solution is the same but with
    :math:`I_\nu \rightarrow J_\nu`.

    """
    np.seterr(over = 'ignore')
    beta = np.sqrt(abs(d2V))
    beta_r = beta*r
    np.seterr(over = 'warn')
    nu = 0.5 * (ndim - 1) # alpha = 2 for S3
    gamma = special.gamma  # Gamma function
    iv, jv = special.iv, special.jv  # (modified) Bessel function
    if beta_r < 1e-2:
        # Use a small-r approximation for the Bessel function.
        s = +1 if d2V > 0 else -1
        phi = 0.0
        dphi = 0.0
        for k in range(1,4):
            _ = (0.5*beta_r)**(2*k-2) * s**k / (gamma(k+1)*gamma(k+1+nu))
            phi += _
            dphi += _ * (2*k)
        phi *= 0.25 * gamma(nu+1) * r**2 * dV * s
        dphi *= 0.25 * gamma(nu+1) * r * dV * s
        phi += phi0
    elif d2V > 0:
        import warnings
        # If beta_r is very large, this will throw off overflow and divide
        # by zero errors in iv(). It will return np.inf though, which is
        # what we want. Just ignore the warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phi = (gamma(nu+1)*(0.5*beta_r)**-nu * iv(nu, beta_r)-1) * dV/d2V
            dphi = -nu*((0.5*beta_r)**-nu / r) * iv(nu, beta_r)
            dphi += (0.5*beta_r)**-nu * 0.5*beta \
                    * (iv(nu-1, beta_r)+iv(nu+1, beta_r))
            dphi *= gamma(nu+1) * dV/d2V
            phi += phi0
    else:
        phi = (gamma(nu+1)*(0.5*beta_r)**-nu * jv(nu, beta_r) - 1) * dV/d2V
        dphi = -nu*((0.5*beta_r)**-nu / r) * jv(nu, beta_r)
        dphi += (0.5*beta_r)**-nu * 0.5*beta \
                * (jv(nu-1, beta_r)-jv(nu+1, beta_r))
        dphi *= gamma(nu+1) * dV/d2V
        phi += phi0
    return phi, dphi

def d2Fdx2(F, x, eps):
    """Calculates `d^2V/dphi^2` using finite differences.
    The finite difference is given by `eps`, and the derivative
    is calculated to fourth order."""
    return (-F(x-2*eps) + 16*F(x-eps) - 30*F(x) + 16*F(x+eps) - F(x+2*eps)) / (12.*eps*eps)


# ==================================================
# Integrators
# ==================================================

class cubicInterpFunction(object):
    """
    Create an interpolating function between two points with a cubic polynomial.

    Like :func:`makeInterpFuncs`, but only uses the first derivatives.
    """
    def __init__(self, y0, dy0, y1, dy1):
        # Easiest to treat this as a bezier curve
        y3 = y1
        y1 = y0 + dy0/3.0
        y2 = y3 - dy1/3.0
        self.Y = y0, y1, y2, y3

    def __call__(self, t):
        mt = 1-t
        y0, y1, y2, y3 = self.Y
        return y0*mt**3 + 3*y1*mt*mt*t + 3*y2*mt*t*t + y3*t**3


def rkqs(y,dydt,t,f, dt_try, epsfrac, epsabs, args=()):
    """
    Take a single 5th order Runge-Kutta step with error monitoring.

    This function is adapted from Numerical Recipes in C.

    The step size dynamically changes such that the error in `y` is smaller
    than the larger of `epsfrac` and `epsabs`. That way, if one wants to
    disregard the fractional error, set `epsfrac` to zero but keep `epsabs`
    non-zero.

    Parameters
    ----------
    y, dydt : array_like
        The initial value and its derivative at the start of the step.
        They should satisfy ``dydt = f(y,t)``. `dydt` is included here for
        efficiency (in case the calling function already calculated it).
    t : float
        The integration variable.
    f : callable
        The derivative function.
    dt_try : float
        An initial guess for the step size.
    epsfrac, epsabs : array_like
        The maximual fractional and absolute errors. Should be either length 1
        or the same size as `y`.
    args : tuple
        Optional arguments for `f`.

    Returns
    -------
    Delta_y : array_like
        Change in `y` during this step.
    Delta_t : float
        Change in `t` during this step.
    dtnext : float
        Best guess for next step size.

    Raises
    ------
    IntegrationError
        If the step size gets smaller than the floating point error.

    References
    ----------
    Based on algorithms described in [1]_.

    .. [1] W. H. Press, et. al. "Numerical Recipes in C: The Art of Scientific
       Computing. Second Edition." Cambridge, 1992.
    """
    dt = dt_try
    while True:
        dy, yerr = rkck(y,dydt,t,f,dt,args)
        errmax = np.nan_to_num(np.max(np.min([abs(yerr/epsabs), abs(yerr)/((abs(y)+1e-300)*epsfrac)], axis=0)))
        if(errmax < 1.0):
            break  # Step succeeded
        dttemp = 0.9*dt*errmax**-.25
        dt = max(dttemp,dt*.1) if dt > 0 else min(dttemp,dt*.1)
        if t + dt == t:
            raise Exception("Stepsize rounds down to zero.")
    if errmax > 1.89e-4:
        dtnext = 0.9 * dt * errmax**-.2
    else:
        dtnext = 5*dt
    return dy, dt, dtnext


def rkck(y,dydt,t,f,dt,args=()):
    """
    Take one 5th-order Cash-Karp Runge-Kutta step.

    Returns
    -------
    array_like
        The change in `y` during this step.
    array_like
        An error estimate for `y`.
    """
    a2=0.2;a3=0.3;a4=0.6;a5=1.0;a6=0.875;b21=0.2  # noqa
    b31=3.0/40.0;b32=9.0/40.0;b41=0.3;b42 = -0.9;b43=1.2;  # noqa
    b51 = -11.0/54.0; b52=2.5;b53 = -70.0/27.0;b54=35.0/27.0;  # noqa
    b61=1631.0/55296.0;b62=175.0/512.0;b63=575.0/13824.0;  # noqa
    b64=44275.0/110592.0;b65=253.0/4096.0;c1=37.0/378.0;  # noqa
    c3=250.0/621.0;c4=125.0/594.0;c6=512.0/1771.0;  # noqa
    dc5 = -277.00/14336.0;  # noqa
    dc1=c1-2825.0/27648.0;dc3=c3-18575.0/48384.0;  # noqa
    dc4=c4-13525.0/55296.0;dc6=c6-0.25  # noqa
    np.seterr(invalid = 'ignore')
    ytemp = y+b21*dt*dydt
    np.seterr(invalid = 'warn')
    ak2 = f(ytemp, t+a2*dt, *args)
    ytemp = y+dt*(b31*dydt+b32*ak2)
    ak3 = f(ytemp, t+a3*dt, *args)
    ytemp = y+dt*(b41*dydt+b42*ak2+b43*ak3)
    ak4 = f(ytemp, t+a4*dt, *args)
    ytemp = y + dt*(b51*dydt+b52*ak2+b53*ak3+b54*ak4)
    ak5 = f(ytemp, t+a5*dt, *args)
    ytemp = y + dt*(b61*dydt+b62*ak2+b63*ak3+b64*ak4+b65*ak5)
    ak6 = f(ytemp, t+a6*dt, *args)
    dyout = dt*(c1*dydt+c3*ak3+c4*ak4+c6*ak6)
    yerr = dt*(dc1*dydt+dc3*ak3+dc4*ak4+dc5*ak5+dc6*ak6)
    return dyout, yerr




# # def findProfile_orig(V, dV, d2V, vev, phi0, N, rmin=1e-6, rmax=1e4,
# #                 rscale=None):
# #     """Shooting method to find the profile.

# #     Parameters
# #     ----------

# #     Returns
# #     ----------

# #     """
    
# #     phitol = 1e-11
# #     xtol   = 1e-11

# #     phibar = findBarrier(V, vev, xtol=1e-15)
# #     if phibar == None:
# #         return np.array([0]), np.zeros((1,2)), "secondOrder" 
    
# #     print(f"Phi barrier = {phibar:2.10g}")
# #     phi_min = phibar
# #     phi_max = vev

# #     xmin = xtol*10
# #     xmax = np.inf
# #     xincrease = 5.0
# #     x = 20
# #     phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
# #     # xmin = 0
# #     # xmax = 1
# #     # x = 0.005  # We expect phi0 to be around 13
# #     # xincrease = 2.0
# #     # phi0 = phi_min + (phi_max - phi_min)*x

# #     # First find Rscale:
# #     # rscale = findProfileScale(V, vev)
# #     # rscale = 0.1*rscale
# #     # rscale = 1e4
# #     # rscale = findProfileScale(V, vev)
# #     # print("rscale_old = ", rscale)
# #     # rscale = findProfileScale_dV(phibar, d2V)
# #     # print("rscale = ", rscale)

    
# #     if rscale == None:
# #         rscale = 10
# #     rmin = rscale * rmin
# #     rmax = rscale * rmax

# #     rmin = 1e-30
    
# #     dr0 = rmin
# #     drmin = dr0 * 0.01

# #     print(f"rmin = {rmin:}")
# #     print(f"rmax = {rmax:}")
# #     print(f"dr0 = {dr0:}")
# #     print(f"drmin = {drmin:}")

# #     # the scale is set by the barrier position no?
# #     # epsabs = abs(np.array([vev, vev/rscale])*phitol)
# #     epsabs = abs(np.array([phibar, phibar/rscale])*phitol)
# #     epsfrac = np.array([1,1]) * phitol

# #     print(f"epsabs = ", epsabs)
# #     print(f"epsfrac = ", epsfrac)

# #     r0 = rmin
# #     # r0, phi0, dphi0 = initialConditions(V, dV, d2V, vev, rmin, phi0, f=1e-4)
# #     phi0, dphi0 = exactSolution(rmin, phi0, dV(phi0), d2V(phi0))

# #     # print(f"Initial conditions: r0 = {r0:2.5g}, phi0 = {phi0:2.5g}, dphi0 = {dphi0:2.5g}")

# #     # get initial conditions
# #     # phi0 = vev * 0.9
# #     # dphi0 = 0.0
# #     y0 = np.array([phi0, dphi0])
# #     r, y, ctype = integrateProfile(r0, y0, dr0, dV, N, epsfrac, epsabs, drmin, rmax)

# #     it = 0
# #     while True:
# #         it += 1
# #         if it > 5000:
# #             break
# #         # print("iteration = ", it)
# #         phi0 = phi_min + (phi_max - phi_min)*np.exp(-x)
# #         # phi0 = phi_min + (phi_max - phi_min)*x
# #         # r0, phi0, dphi0 = initialConditions(V, dV, d2V, vev, rmin, phi0, f=1e-4)
# #         phi0, dphi0 = exactSolution(rmin, phi0, dV(phi0), d2V(phi0))
# #         y0 = np.array([phi0, dphi0])
# #         if not np.isfinite(r0) or not np.isfinite(x):
# #             # Use the last finite values instead
# #             # (assuming there are such values)
# #             print("fail!!")
# #             assert r is not None, "Failed to retrieve initial "\
# #                 "conditions on the first try."
# #             break
# #         # print("#"*50)
# #         # print(f"Initial conditions: r0 = {r0:2.5g}, phi0 = {phi0:2.5g}, dphi0 = {dphi0:2.5g}")
# #         r, y, ctype = integrateProfile(r0, y0, dr0, dV, N, epsfrac, epsabs, drmin, rmax)
# #         # print("ctype = ", ctype)
# #         # print(f"x = {x:2.5g}")
# #         # print(f"xmin = {xmin:2.5g}")
# #         # print(f"xmax = {xmax:2.5g}")
# #         # print(f"phi0 = {phi0:2.5g}")
# #         if ctype == "converged":
# #             print("Converged !!!")
# #             break
# #         # elif ctype == "overflow" or ctype == "overshoot": # x is too small!
# #         elif ctype == "undershoot":  # x is too big!
# #             xmax = x
# #             x = .5*(xmin+xmax)
# #             # print(f"Overshoot: setting xmax to {xmax:2.10g}")
# #         # elif ctype == "undershoot":  # x is too big!
# #         elif ctype == "overflow" or ctype == "overshoot": # x is too small!
# #             xmin = x
# #             # print(f"Undershoot: setting xmin to {xmin:2.10g}")
# #             x = x*xincrease if xmax == np.inf else .5*(xmin+xmax)
# 
#         # Reached tolerance?
#         if (xmax-xmin) < xtol:
#             print("Reached tolerance")
#             break

#     y = np.asarray(y)
#     r = np.asarray(r)

#     return r, y, ctype

