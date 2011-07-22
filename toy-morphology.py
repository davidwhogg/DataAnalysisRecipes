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
size_sigma = 0.05

def get_integral(alpha, S):
    return S**(1. - alpha) / (1. - alpha)

def integrate_over_bin(alpha, Sa, Sb):
    return get_integral(alpha, Sa) - get_integral(alpha, Sb)

def make_flux(alpha, Shi, Slo, size=(1)):
    return ((1. - alpha) * (np.random.uniform(size=size) * integrate_over_bin(alpha, Shi, Slo) + get_integral(alpha, Slo)))**(1. / (1. - alpha))

def observed_size(truesize):
    return np.sqrt(truesize**2 + 1.0) + size_sigma * np.random.normal(size=truesize.shape)

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

def quantiles(x):
    nx = len(x)
    I = np.argsort(x)
    q = np.zeros(nx).astype(int)
    for i in range(4):
        q[I[(i * nx / 4):((i + 1) * nx / 4)]] = i
    splits = np.zeros(3).astype(float)
    for i in range(3):
        splits[i] = 0.5 * (np.max(x[q==i]) + np.min(x[q==(i+1)]))
    return q, splits

def straight_cut(t, c, ntotal=None):
    nc = len(c)
    cuts = np.arange(0.9,1.4,0.01)
    if ntotal is None:
        ntotal = np.sum(c == 0)
    completeness = np.array([float(np.sum(c[t < cut] == 0)) / float(ntotal) for cut in cuts])
    purity = np.array([float(np.sum(c[t < cut] == 0)) / float(np.sum(c[t < cut] < 3)) for cut in cuts])
    return cuts, completeness, purity

def prob_cut(logr, c):
    nc = len(c)
    cuts = np.arange(-5., 5., 0.1)
    completeness = np.array([float(np.sum(c[logr > cut] == 0)) / float(np.sum(c == 0)) for cut in cuts])
    purity = np.array([float(np.sum(c[logr > cut] == 0)) / float(np.sum(c[logr > cut] < 3)) for cut in cuts])
    return cuts, completeness, purity

def Gaussian(x, m, sigma):
    return np.exp(-0.5 * (x - m)**2 / sigma**2) / np.sqrt(2. * np.pi * sigma**2)

def likelihood_star(t):
    return Gaussian(t, 1.0, size_sigma)

np.random.seed(21)
gamma_sample = np.random.gamma(3., size=20000)
gamma_weight = np.ones_like(gamma_sample)
gamma_weight /= np.sum(gamma_weight)

# assumes the t, m come in one gala at a time
# scale is the scale of the size distribution, not of an individual galaxy
def likelihood_gala(ts, scale):
    return np.array([np.sum(gamma_weight * Gaussian(t, np.sqrt(1. + (scale * gamma_sample)**2), size_sigma)) for t in ts])

def probability(ts, pstar, scale):
    lstar = likelihood_star(ts)
    lgala = likelihood_gala(ts, scale)
    return pstar * lstar + (1. - pstar) * lgala

def total_log_probability(ts, pstar, scale):
    return np.sum(np.log(probability(ts, pstar, scale)))

def cost(pars, args):
    logodds, scale = pars
    pstar = np.exp(logodds) / (1. + np.exp(logodds))
    print pstar, scale
    ts = args
    return -1. * total_log_probability(ts, pstar, scale)

def main():
    np.random.seed(42)

    mstar, tstar, mgala, tgala = make_truth()
    cstar = np.zeros_like(mstar).astype(int)
    cgala = np.ones_like(mgala).astype(int)
    m = np.append(mstar, mgala)
    t = np.append(tstar, tgala)
    c = np.append(cstar, cgala)

    plt.clf()
    plt.hist(mgala, bins=5, histtype='step', color='b')
    plt.hist(mstar, bins=5, histtype='step', color='g')
    plt.xlim(20., 25.)
    plt.xlabel('magnitude $m$')
    plt.semilogy()
    plt.savefig('%s-hist.%s' % (prefix, suffix))

    plt.clf()
    plt.plot(tgala, mgala, 'bo', mew=0, alpha=0.5)
    plt.plot(tstar, mstar, 'go', mew=0, alpha=0.5)
    sizelim = (0.0, 10.0)
    plt.xlim(sizelim)
    plt.xlabel(r'size $\theta$')
    maglim = (25., 20.)
    plt.ylim(maglim)
    plt.ylabel('magnitude $m$')
    plt.savefig('%s-labeled-data.%s' % (prefix, suffix))

    plt.clf()
    plt.plot(t, m, 'ko', mew=0, alpha=0.25)
    plt.xlim(sizelim)
    plt.xlabel(r'size $\theta$')
    plt.ylim(maglim)
    plt.ylabel('magnitude $m$')
    plt.savefig('%s-data.%s' % (prefix, suffix))
    quants, splits = quantiles(m)
    for split in splits:
        plt.axhline(split, color='r', lw=2., alpha=0.75)
    plt.savefig('%s-data-qs.%s' % (prefix, suffix))

    lstar = likelihood_star(t)
    lgala = np.zeros_like(lstar)
    pstar = np.zeros_like(lstar)
    pgala = np.zeros_like(lstar)

    pstarbest = np.array([])
    scalebest = np.array([])
    lo, scale = 0.0, 1.0
    for q in range(4):
        I = (quants == q)
        tfit = t[I]
        lo, scale = op.fmin(cost, (lo, scale), args=(tfit,))
        thispstar = np.exp(lo) / (1. + np.exp(lo))
        pstarbest = np.append(pstarbest, thispstar)
        scalebest = np.append(scalebest, scale)
        lgala[I] = likelihood_gala(t[I], scale)
        pstar[I] = thispstar * lstar[I]
        pgala[I] = (1. - thispstar) * lgala[I]
        tplot = np.arange(0.8, 10.0, 0.01)
        pplot = probability(tplot, thispstar, scale)
        if q < 3:
            baseline = splits[q]
        else:
            baseline = 25.
        plt.plot(tplot, baseline - 0.7 * pplot / np.max(pplot), 'r-', lw=2., alpha=0.75)
    plt.savefig('%s-data-models.%s' % (prefix, suffix))
    print pstarbest, scalebest

    plt.clf()
    plt.plot(t, m, 'ko', mew=0, alpha=0.5)
    plt.xlim(sizelim)
    plt.xlabel(r'size $\theta$')
    plt.ylim(maglim)
    plt.ylabel('magnitude $m$')
    excut = 1.1
    plt.axvline(excut, color='r', lw=2, alpha=0.75)
    plt.savefig('%s-data-cut.%s' % (prefix, suffix))

    cuts, completeness, purity = straight_cut(t, c)
    plt.clf()
    plt.plot(cuts, completeness, 'k-')
    plt.plot(cuts, purity, 'k--')
    plt.xlim(np.min(cuts), np.max(cuts))
    plt.xlabel(r'size cut $\theta_c$')
    plt.ylim(0., 1.)
    plt.title('hard cut: completeness and purity')
    plt.savefig('%s-hard.%s' % (prefix, suffix))

    plt.clf()
    plt.plot(t, m, 'ko', mew=0, alpha=0.5)
    plt.xlim(sizelim)
    plt.xlabel(r'size $\theta$')
    plt.ylim(maglim)
    plt.ylabel('magnitude $m$')
    plt.plot([excut, excut, -1.], [0., 24., 24.], 'r-', lw=2, alpha=0.75)
    plt.savefig('%s-data-cut-24.%s' % (prefix, suffix))

    cuts, completeness, purity = straight_cut(t[m < 24.], c[m < 24.], ntotal=np.sum(c==0))
    plt.clf()
    plt.plot(cuts, completeness, 'k-')
    plt.plot(cuts, purity, 'k--')
    plt.xlim(np.min(cuts), np.max(cuts))
    plt.xlabel(r'size cut $\theta_c$')
    plt.ylim(0., 1.)
    plt.title('hard cut: completeness and purity for $m < 24$')
    plt.savefig('%s-hard-24.%s' % (prefix, suffix))

    cuts, completeness, purity = prob_cut(np.log(lstar / lgala), c)
    plt.clf()
    plt.plot(cuts, completeness, 'k-')
    plt.plot(cuts, purity, 'k--')
    plt.xlim(np.min(cuts), np.max(cuts))
    plt.xlabel('ln likelihood ratio cut')
    plt.ylim(0., 1.)
    plt.title('likelihood ratio cut: completeness and purity')
    plt.savefig('%s-like.%s' % (prefix, suffix))

    cuts, completeness, purity = prob_cut(np.log(pstar / pgala), c)
    plt.clf()
    plt.plot(cuts, completeness, 'k-')
    plt.plot(cuts, purity, 'k--')
    plt.xlim(np.min(cuts), np.max(cuts))
    plt.xlabel('ln probability ratio cut')
    plt.ylim(0., 1.)
    plt.title('probability ratio cut: completeness and purity')
    plt.savefig('%s-prob.%s' % (prefix, suffix))

if __name__ == '__main__':
    main()
