"""Module to compute the potential and transition rate of
the RS radion potential.

Author: Jonas Matuszak <jonas.matuszak@kit.edu>, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import interpolate

import calcBounce as cb


class Potential():
    def __init__(self, xmin, vir, epsilon, delta, n, N=4.5, withQCD=True):
        """Radion potential.

        Parameters
        ----------

        Returns
        ----------

        """
        self.xmin = xmin
        self.vir = vir
        self.epsilon = epsilon
        self.delta = delta
        self.N = N
        self.n = n
        self.withQCD = withQCD

        # QCD constants
        # self.LQCD0 = 0.09
        self.LQCD0 = 0.09
        self.nc = 3
        self.bQCD = 7
        self.xc = xmin * (self.LQCD0/(self.nc * xmin))**(1/(1-n))

        self.Mpl_GeV = 1.220910e19

        self.Xmin = 1/(1 + epsilon/2.) * \
            (1 + epsilon/4. + np.sign(epsilon)/2. * np.sqrt(epsilon + epsilon**2/4. - delta/vir**2))

        self.VQCD0 = self.VQCD(1e-100)

        # self.xbar0 = self.findBarrier(T=0)
        # self.xmeta_min = self.findMetaMin(self.xbar0)


    def VAdS(self, Th, T):
        """description

        Parameters
        ----------

        Returns
        ----------

        """
        V = 3/8. * np.pi**2 * self.N**2 * Th**4
        V -= 0.5 * np.pi**2 * self.N**2 * Th**3 * T
        return V

    def dVAdS(self, Th, T):
        """description

        Parameters
        ----------

        Returns
        ----------

        """
        dV = 12/8. * np.pi**2 * self.N**2 * Th**3
        dV -= 1.5 * np.pi**2 * self.N**2 * Th**2 * T
        return -dV

    def d2VAdS(self, Th, T):
        """description

        Parameters
        ----------

        Returns
        ----------

        """
        d2V = 3*12/8. * np.pi**2 * self.N**2 * Th**2
        d2V -= 3 * np.pi**2 * self.N**2 * Th * T
        return d2V

    def VGW(self, x):
        V = x**4 * self.vir**2 * ((4+2*self.epsilon)*(1 - self.Xmin * (x/self.xmin)**self.epsilon)**2
                                  - self.epsilon + self.delta/self.vir**2)
        return V

    def dVGW(self, x):
        x = x + 0j
        res = self.vir**2 * x**3 * \
            (4*self.delta/self.vir**2 + (4+self.epsilon)* \
             (4*(-1 + (x/self.xmin)**self.epsilon)**2 +
              self.epsilon*(-2 + (x/self.xmin)**self.epsilon) *(x/self.xmin)**self.epsilon) +
             2*(4+self.epsilon)*np.sqrt(4*self.epsilon + self.epsilon**2 - 4*self.delta/self.vir**2)*
             (-1 + (x/self.xmin)**self.epsilon)*(x/self.xmin)**self.epsilon * np.sign(self.epsilon) +
             (-4*self.delta + self.epsilon*(4+self.epsilon)*self.vir**2)*(x/self.xmin)**(2*self.epsilon)/self.vir**2)

        res = np.real(res)    
        return res

    def d2VGW(self, x):
        """Second derivative of the Goldberger-Wise potential

        Parameters
        ----------

        Returns
        ----------

        """
        res = 4*self.vir**2*x**2 * \
            (self.Xmin*self.epsilon**3*(x/self.xmin)**self.epsilon*(2*self.Xmin*(x/self.xmin)**self.epsilon - 1) +
             self.Xmin*self.epsilon**2*(x/self.xmin)**self.epsilon*(11*self.Xmin*(x/self.xmin)**self.epsilon - 9) +
             3*self.delta/self.vir**2 + self.epsilon*(20*self.Xmin**2*(x/self.xmin)**(2*self.epsilon) -
                                   26*self.Xmin*(x/self.xmin)**self.epsilon + 3) +
             12*(self.Xmin*(x/self.xmin)**self.epsilon - 1)**2)

        return res

    def LQCDapprox(self, x):
        """Smoothed QCD scale."""
        return self.LQCD0 * ((x*np.exp(-(self.xc/(x+1e-100))**2) +
                              self.xc)/self.xmin)**self.n

    def VQCD(self, x):
        """QCD potential"""
        res = - self.bQCD / 17.0 * self.LQCDapprox(x)**4
        return res

    def dVQCD(self, x):
        """Derivative of the QCD potentital

        Parameters
        ----------

        Returns
        ----------

        """
        x = np.abs(x)
        res = - 4 * self.bQCD * self.n * self.LQCD0**4 / (17 * self.xmin)
        res = res * np.power((np.exp(-(self.xc/(x+1e-50))**2) * x + self.xc)/self.xmin, 4*self.n-1)
        res = res * (np.exp(-(self.xc/(x+1e-50))**2) + 2 * \
                         np.exp(-(self.xc/(x+1e-50))**2) * self.xc**2 / x**2)
        return res

    def d2VQCD(self, x):
        """"""
        x = x + 1e-50
        expterm = np.exp(-(self.xc/x)**2)
        res = -self.LQCD0**4 * self.bQCD* \
            (4*self.n*((x*expterm + self.xc)/self.xmin)**(4*self.n - 1) *
             (-2*self.xc**2*expterm/x**3 + 4*self.xc**4*expterm/x**5)/self.xmin
             + 4*self.n*((x*expterm + self.xc)/self.xmin)**(4*self.n - 2) *
             (4*self.n - 1)*(expterm + 2*self.xc**2*expterm/x**2)**2/self.xmin**2)/17.0
        return res

    def Vfull(self, x, T):
        """description

        Parameters
        ----------

        Returns
        ----------

        """
        if np.asarray(x).ndim == 0:
            if x <= 0:
                Th = -x
                return self.VAdS(Th, T)
            else:
                mu = x
                Vc = self.VGW(mu)
                if self.withQCD:
                    Vc += self.VQCD(mu) - self.VQCD0
                return Vc
        Vf = np.zeros_like(x)
        Th = - x[np.where(x <= 0)]
        Vf[np.where(x <=0)] = self.VAdS(Th, T)
        mu = x[np.where(x > 0)]
        Vf[np.where(x > 0)] = self.VGW(mu)
        if self.withQCD:
            VQCD = self.VQCD(mu) - self.VQCD0
            Vf[np.where(x > 0)] += VQCD
            # for i, m in zip(np.where(x > 0), mu):
            #     Vf[i] += self.VQCD(m) - self.VQCD0
        return Vf

    def dVfull(self, x, T):
        if np.asarray(x).ndim == 0:
            if x <= 0:
                Th = - x
                return self.dVAdS(Th, T)
            else:
                mu = x
                dVc = self.dVGW(mu)
                if self.withQCD:
                    dVc += self.dVQCD(mu)
                return dVc
        dVf = np.zeros_like(x)
        Th = -x[np.where(x <= 0)]
        dVf[np.where(x <= 0)] = self.dVAdS(Th, T)
        mu = x[np.where(x > 0)]
        dVf[np.where(x > 0)] = self.dVGW(mu)
        if self.withQCD:
            dVf[np.where(x > 0)] += self.dVQCD(mu)
        return dVf

    def d2Vfull(self, x, T):
        if np.asarray(x).ndim == 0:
            if x <= 0:
                Th = -x
                return self.d2VAdS(Th, T)
            else:
                mu = x
                d2Vc = self.d2VGW(mu)
                if self.withQCD:
                    d2Vc += self.d2VQCD(mu)
                return d2Vc
        d2Vf = np.zeros_like(x)
        Th = -x[np.where(x <= 0)]
        d2Vf[np.where(x <= 0)] = self.d2VAdS(Th, T)
        mu = x[np.where(x > 0)]
        d2Vf[np.where(x > 0)] = self.d2VGW(mu)
        if self.withQCD:
            d2Vf[np.where(x > 0)] += self.d2VQCD(mu)
        return d2Vf

    def findBarrier(self, T: float) -> float:
        """Find the location of the potential barrier."""
        res = optimize.minimize_scalar(lambda x: -self.Vfull(x, T=T), bounds=(-T, self.xmin),
                                       options={"xatol": 1e-20})
        if not res.success:
            raise Exception("Did not find potential barrier!")
        xbar = res.x
        return xbar

    def findMetaMin(self, xbar: float, xmin=0.0) -> float:
        """Find the unstable minimum, it can be non-zero with QCD effects!

        Parameters
        ----------
        xbar : float
        xmin : float

        Returns
        ----------
        xmeta_min
        """
        res = optimize.minimize_scalar(self.Vfull, bounds=(xmin, xbar), args=(0),
                                       options={"xatol": 1e-20})
        if not res.success:
            xmeta_min = 0.0
        else:
            xmeta_min = res.x
        return xmeta_min

    def plotPotential(self, xmax: float, T: float, xmin=0):
        """Plot the potential.

        Parameters
        ----------

        Returns
        ----------

        """
        xrange = np.linspace(xmin, xmax, 200)
        plt.plot(xrange, self.Vfull(xrange, T), label=r"$V_{\mathrm{GW}} + V_{\mathrm{QCD}}$")
        plt.plot(xrange, self.VGW(xrange), label=r"$V_{\mathrm{GW}}$", ls="-.")
        plt.plot(xrange, self.VQCD(xrange) - self.VQCD0, label=r"$V_{\mathrm{QCD}}$", ls="--")
        plt.legend()
        plt.show()


    def plotPotentialAll(self, xmax, T, xmin=0):
        """Plot the potential and its derivatives.

        Parameters
        ----------

        Returns
        ----------

        """
        xbar = self.findBarrier(T)
        xrange = np.linspace(xmin, xmax, 200)
        fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        ax[0].plot(xrange, self.Vfull(xrange, T), label=r"$V_{\mathrm{AdS}} \mathrm{~and~}V_{\mathrm{GW}} + V_{\mathrm{QCD}}$")
        # ax[0].plot(xrange, self.VGW(xrange), label=r"$V_{\mathrm{GW}}$", ls="-.")
        # ax[0].plot(xrange, self.VQCD(xrange) - self.VQCD0, label=r"$V_{\mathrm{QCD}}$", ls="--")
        
        ax[0].set_ylabel(r"$V$")
        ax[0].legend()

        ax[1].plot(xrange, self.dVfull(xrange, T), label=r"$V_{\mathrm{GW}} + V_{\mathrm{QCD}}$")
        # ax[1].plot(xrange, self.dVGW(xrange), label=r"$V_{\mathrm{GW}}$", ls="-.")
        # ax[1].plot(xrange, self.dVQCD(xrange) - self.VQCD0, label=r"$V_{\mathrm{QCD}}$", ls="--")
        ax[1].set_ylabel(r"$\partial_\phi V$")
        ax[1].legend()
        
        
        ax[2].plot(xrange, self.d2Vfull(xrange, T), label=r"$V_{\mathrm{GW}} + V_{\mathrm{QCD}}$")
        # ax[2].plot(xrange, self.d2VGW(xrange), label=r"$V_{\mathrm{GW}}$", ls="-.")
        # ax[2].plot(xrange, self.d2VQCD(xrange), label=r"$V_{\mathrm{QCD}}$", ls="--")
        
        ax[2].set_ylabel(r"$\partial^2_\phi V$")
        ax[2].legend()

        if xbar < xmax:
            ax[0].plot(xbar, self.Vfull(xbar, T), 'o')
            ax[1].plot(xbar, self.dVfull(xbar, T), 'o')
            ax[2].plot(xbar, self.d2Vfull(xbar, T), 'o')
        ax[0].set_title(r"$v_{\mathrm{IR}} = " + f"{self.vir:2.5g}" + r"~~\epsilon = " + f"{self.epsilon:2.5g}$")
        
        plt.show()
        
