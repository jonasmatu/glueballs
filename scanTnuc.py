"""Module to scan over the parameter space to find the nucleation temperature

Author: Jonas Matuszak <jonas.matuszak@kit.edu>, 2025
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
import h5py
import argparse
import os
import yaml

from scipy import optimize
from scipy import interpolate

import calcBounce as cb
from calc_nucleation import findNucleationTemp, findNucleationTempTriangleApprox
from potential import Potential



def getTnuc(pot, Tmax, Tmin, approx=False):
    try:
        if not approx:
            Tnuc3 = findNucleationTemp(pot, Tmax, Tmin, ndim=3)
        else:
            Tnuc3 = findNucleationTempTriangleApprox(pot, Tmax, Tmin, ndim=3)
    except:
        Tnuc3 = -1
    try:
        if not approx:
            Tnuc4 = findNucleationTemp(pot, Tmax, Tmin, ndim=4)
        else:
            Tnuc4 = findNucleationTempTriangleApprox(pot, Tmax, Tmin, ndim=4)
    except:
        Tnuc4 = -1

    return Tnuc3, Tnuc4


def scanTnuc(fname, xmin, deltat, n, N, withQCD=True, npoints=30, n_jobs=12,
             approx=False):
    # Reproduce fig. 4 of servant and harling
    vir_range = np.linspace(0.0, 1.0, npoints)
    eps_range = np.linspace(0.01, .1, npoints)

    Tnuc3 = np.zeros((npoints, npoints))
    Tnuc4 = np.zeros((npoints, npoints))

    pool = multiprocessing.Pool(processes=n_jobs)
    input_params = []
    for i, vir in enumerate(vir_range):
        for j, eps in enumerate(eps_range):
            delta = vir**2 * deltat
            pot = Potential(xmin, vir, eps, delta, n, N=N, withQCD=withQCD)
            Tcrit = np.power(-8* pot.VGW(xmin)/(np.pi**2 * N**2), 1/4.0)
            # by hand: 
            Tmax = Tcrit*0.99
            # Tmax = 50
            input_params.append((pot, Tmax, 1e-10, approx))

    results = pool.starmap(getTnuc, input_params)
    pool.close()
    for i in range(npoints):
        for j in range(npoints):
            datapoint = results[i*npoints + j]
            Tnuc3[i, j], Tnuc4[i, j] = datapoint

    # Backup in case i made a mistake in h5py code
    npfname = fname.split(".")[0] + ".npy"
    np.save(npfname, np.array([Tnuc3, Tnuc4]))

    with h5py.File(fname, "w") as f:
        g = f.create_group("data")
        g.attrs["npoints"] = npoints
        g.attrs["vir_start"], g.attrs["vir_end"] = vir_range[0], vir_range[-1]
        g.attrs["eps_start"], g.attrs["eps_end"] = eps_range[0], eps_range[-1]
        g.attrs["xmin"] = xmin
        g.attrs["deltat"] = deltat
        g.attrs["n"] = n
        g.attrs["N"] = N
        g.attrs["withQCD"] = withQCD
        g["data/Tnuc3"] = Tnuc3
        g["data/Tnuc4"] = Tnuc4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter scan for conformal model.')
    parser.add_argument('-j', '--jobs', type=int)
    parser.add_argument('-f', '--folder', type=str)
    parser.add_argument('-v', '--verbose', type=str)

    args = parser.parse_args()

    # ==================================================
    # Initialise scan with parameters from config.yaml
    # ==================================================
    folder = args.folder
    if folder[-1] != "/":
        folder += "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder + "config.yaml", 'r') as f:
        yamldata = yaml.safe_load(f)

    config = yamldata["config"]

    npoints = config["npoints"]
    approx = config["approx"]
    xmin = config["xmin"]
    withQCD = config["withQCD"]
    n = config["n"]
    N = config["N"]
    deltat = config["deltat"]
        
    fname = folder + "data.hdf5"
    scanTnuc(fname, xmin, deltat, n, N, withQCD=withQCD,
             npoints=npoints, n_jobs=args.jobs, approx=approx)
