"""

"""

import numpy as np
import matplotlib.pyplot as plt


from potential import RadionPotential
from calcBounce import *



xmin = 2.5e3
N = 4.5
n = 0.3
deltat = -0.5


vir = 0.5

delta = deltat * vir**2


#epsilon = 0.033

eps_range = np.linspace(0,0.1, 5)

for i, epsilon in enumerate(eps_range):
    
    print("#"*50)
    print(f"epsilon = {epsilon:2.5g}")
    
    pot = RadionPotential(xmin, vir, epsilon, delta, n, N, withQCD=True)
    # pot.plotPotentialAll(1e-6)

    rscale = 1

    r, y, ctype = findProfile(pot.Vfull, pot.dVfull, pot.d2Vfull, xmin,
                              xmin*1e-4, N=4.5, rmin=1e-6, rmax=1e12,
                              rscale=rscale)


    phi, dphi = y.T
    phi0 = phi[0]
    S = findActionRadion(r, phi, dphi, N, pot.Vfull)

    print(f"Action = {S:2.5g}, phi[-1] = {phi[-1]:2.5g}")
    print("\n")
    Swired = findActionRadion(r, phi, dphi, N, pot.Vfull)
    
    plt.semilogx(r, phi)
    plt.show()
