"""Module to compute the potential and transition rate of
the RS radion potential.

Author: Jonas Matuszak <jonas.matuszak@kit.edu>, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import interpolate

import calcBounce as cb

import warnings
# warnings.filterwarnings("error")


class RadionPotential():
    def __init__(self, xmin, vir, epsilon, delta, n, N=4.5, withQCD=False):
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

        self.xbar = self.findBarrier()
        self.xmeta_min = self.findMetaMin(self.xbar)


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

    def Vfull(self, x):
        V = self.VGW(x)
        if self.withQCD:
            V += self.VQCD(x) - self.VQCD0
        return V

    def dVfull(self, x):
        dV = self.dVGW(x)
        if self.withQCD:
            dV += self.dVQCD(x)
        return dV

    def d2Vfull(self, x):
        d2V = self.d2VGW(x)
        if self.withQCD:
            d2V += self.d2VQCD(x)
        return d2V

    def findBarrier(self):
        """Find the location of the potential barrier

        Parameters
        ----------
        
        Returns
        ----------
        xbar : float
        """
        res = optimize.minimize_scalar(lambda x: -self.Vfull(x), bounds=(0, self.xmin), method='bounded',
                                       tol=1e-15)
        if not res.success:
            raise Exception("Did not find potential barrier!")
        xbar = res.x
        return xbar

    def findMetaMin(self, xbar: float) -> float:
        """Find the unstable minimum, it can be non-zero with QCD effects!

        Parameters
        ----------
        xbar : float

        Returns
        ----------
        xmeta_min
        """
        res = optimize.minimize_scalar(self.Vfull, bounds=(0, xbar), tol=1e-20)
        if not res.success:
            xmeta_min = 0.0
        else:
            xmeta_min = res.x
        return xmeta_min

    def plotPotential(self, xmax, xmin=0):
        """Plot the potential.

        Parameters
        ----------

        Returns
        ----------

        """
        xrange = np.linspace(xmin, xmax, 200)
        plt.plot(xrange, self.Vfull(xrange), label=r"$V_{\mathrm{GW}} + V_{\mathrm{QCD}}$")
        plt.plot(xrange, self.VGW(xrange), label=r"$V_{\mathrm{GW}}$", ls="-.")
        plt.plot(xrange, self.VQCD(xrange) - self.VQCD0, label=r"$V_{\mathrm{QCD}}$", ls="--")
        plt.legend()
        plt.show()


    def plotPotentialAll(self, xmax, xmin=0):
        """Plot the potential and its derivatives.

        Parameters
        ----------

        Returns
        ----------

        """
        xbar = cb.findBarrier(self.Vfull, self.xmin, xtol=1e-15)
        xrange = np.linspace(xmin, xmax, 200)
        fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        ax[0].plot(xrange, self.Vfull(xrange), label=r"$V_{\mathrm{GW}} + V_{\mathrm{QCD}}$")
        ax[0].plot(xrange, self.VGW(xrange), label=r"$V_{\mathrm{GW}}$", ls="-.")
        ax[0].plot(xrange, self.VQCD(xrange) - self.VQCD0, label=r"$V_{\mathrm{QCD}}$", ls="--")
        
        ax[0].set_ylabel(r"$V$")
        ax[0].legend()

        ax[1].plot(xrange, self.dVfull(xrange), label=r"$V_{\mathrm{GW}} + V_{\mathrm{QCD}}$")
        ax[1].plot(xrange, self.dVGW(xrange), label=r"$V_{\mathrm{GW}}$", ls="-.")
        ax[1].plot(xrange, self.dVQCD(xrange) - self.VQCD0, label=r"$V_{\mathrm{QCD}}$", ls="--")
        ax[1].set_ylabel(r"$\partial_\phi V$")
        ax[1].legend()
        
        
        ax[2].plot(xrange, self.d2Vfull(xrange), label=r"$V_{\mathrm{GW}} + V_{\mathrm{QCD}}$")
        ax[2].plot(xrange, self.d2VGW(xrange), label=r"$V_{\mathrm{GW}}$", ls="-.")
        ax[2].plot(xrange, self.d2VQCD(xrange), label=r"$V_{\mathrm{QCD}}$", ls="--")
        
        ax[2].set_ylabel(r"$\partial^2_\phi V$")
        ax[2].legend()

        if xbar < xmax:
            ax[0].plot(xbar, self.Vfull(xbar), 'o')
            ax[1].plot(xbar, self.dVfull(xbar), 'o')
            ax[2].plot(xbar, self.d2Vfull(xbar), 'o')
        ax[0].set_title(r"$v_{\mathrm{IR}} = " + f"{self.vir:2.5g}" + r"~~\epsilon = " + f"{self.epsilon:2.5g}$")
        
        plt.show()
        
