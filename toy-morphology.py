# Copyright 2011 David W. Hogg (NYU).
# All rights reserved.

# Code to make a toy model of morphological star-galaxy separation

import numpy as np
# this rc block must be before the matplotlib import?
from matplotlib import rc
rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':18})
rc('text', usetex=True)
# now import matplotlib
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import scipy.optimize as op

prefix = 'toy-morph'
suffix = 'png'

def get_integral(alpha, S):
    return S**(1. - alpha) / (1. - alpha)

def integrate_over_bin(alpha, Sa, Sb):
    return get_integral(alpha, Sa) - get_integral(alpha, Sb)

def make_flux(alpha, Shi, Slo, size=(1)):
    return ((1. - alpha) * (np.random.uniform(size=size) * integrate_over_bin(alpha, Shi, Slo) + get_integral(alpha, Slo)))**(1. / (1. - alpha))

def observed_size(truesize):
    return np.sqrt(truesize**2 + 1.0) + 0.05 * np.random.normal(size=truesize.shape)

def mag_from_flux(flux):
    return 25. - 2.5 * np.log(flux) / np.log(10.)

def make_truth():
    Slo = 1.0
    Shi = 100.0
    nstar = 100
    ngala = 1000
    alphastar = 1.1
    alphagala = 1.7
    fluxstar = make_flux(alphastar, Slo, Shi, size=nstar)
    fluxgala = make_flux(alphagala, Slo, Shi, size=ngala)
    magstar = mag_from_flux(fluxstar)
    maggala = mag_from_flux(fluxgala)
    truesizestar = np.zeros_like(magstar)
    truesizegala = 0.4 * (25.5 - maggala) * np.random.gamma(3., size=maggala.shape)
    sizestar = observed_size(truesizestar)
    sizegala = observed_size(truesizegala)
    return magstar, sizestar, maggala, sizegala

if __name__ == '__main__':
    np.random.seed(42)
    mstar, tstar, mgala, tgala = make_truth()
    m = np.append(mstar, mgala)
    t = np.append(tstar, tgala)
    plt.clf()
    plt.hist(mgala, bins=5, histtype='step', color='b', alpha=0.5)
    plt.hist(mstar, bins=5, histtype='step', color='g', alpha=0.5)
    plt.xlim(20., 25.)
    plt.semilogy()
    plt.savefig('%s-hist.%s' % (prefix, suffix))
    plt.clf()
    plt.plot(tgala, mgala, 'bo', mew=0, alpha=0.5)
    plt.plot(tstar, mstar, 'go', mew=0, alpha=0.5)
    plt.xlim(0.0, 10.0)
    plt.ylim(25., 20.)
    plt.savefig('%s-labeled-data.%s' % (prefix, suffix))
    plt.clf()
    plt.plot(t, m, 'ko', mew=0, alpha=0.5)
    plt.xlim(0.0, 10.0)
    plt.ylim(25., 20.)
    plt.savefig('%s-data.%s' % (prefix, suffix))
