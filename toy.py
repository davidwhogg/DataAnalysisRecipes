# Copyright 2011 David W. Hogg (NYU).
# All rights reserved.

# Code to make a toy model that demonstrates power of hierarchical modeling.

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

def make_truth():
    nstar = 100
    ngalaxy = 1000
    ystar = np.random.normal(1.2, 1.0, size=nstar)
    xstar = 0.1 * ystar**2 + 0.02 * np.random.normal(size=nstar)
    cstar = np.zeros(nstar).astype(int)
    xgalaxy = np.random.normal(1.2, 1.0, size=ngalaxy)
    ygalaxy = 0.1 * (xgalaxy - 0.5)**2 + 0.02 * np.random.normal(size=ngalaxy)
    cgalaxy = np.ones(ngalaxy).astype(int)
    x = np.append(xstar, xgalaxy)
    y = np.append(ystar, ygalaxy)
    c = np.append(cstar, cgalaxy)
    return ((x, y), c)

def noisify(xy, c):
    nobj = len(c)
    x,y = xy
    invvar = 1.0 / (0.4 * np.random.uniform(size=nobj))**2
    nx = x + np.random.normal(size=nobj)/np.sqrt(invvar)
    ny = y + np.random.normal(size=nobj)/np.sqrt(invvar)
    bad = (np.random.uniform(size=nobj) < 0.01)
    nc = np.zeros(nobj)
    nc[:] = c[:]
    nc[bad] = 1 - nc[bad]
    return ((nx, ny), nc, invvar)

def make_models():
    ystar = np.arange(-1., 2., 0.05)
    xstar = 0.15 * ystar**2 + 0.2 * np.random.normal(size=len(ystar))
    ystar2 = np.arange(-1., 4., 0.2)
    xstar2 = 0.1 * ystar2**2 + 0.02 * np.random.normal(size=len(ystar2))
    ystar = np.append(ystar, ystar2)
    xstar = np.append(xstar, xstar2)
    cstar = np.zeros(len(ystar)).astype(int)
    xgalaxy = np.arange(-1., 2., 0.05)
    ygalaxy = 0.15 * (xgalaxy - 0.45)**2 + 0.2 * np.random.normal(size=len(xgalaxy))
    xgalaxy2 = np.arange(-1., 4., 0.2)
    ygalaxy2 = 0.1 * (xgalaxy2 - 0.5)**2 + 0.02 * np.random.normal(size=len(xgalaxy2))
    xgalaxy = np.append(xgalaxy, xgalaxy2)
    ygalaxy = np.append(ygalaxy, ygalaxy2)
    cgalaxy = np.ones(len(ygalaxy)).astype(int)
    xmod = np.append(xstar, xgalaxy)
    ymod = np.append(ystar, ygalaxy)
    cmod = np.append(cstar, cgalaxy)
    return ((xmod, ymod), cmod)

# serious duck typing in this function
def ln_likelihood(x, y, invvar, modelxy):
    mx, my = modelxy
    return -0.5 * invvar * ((x - mx)**2 + (y - my)**2)

def ml_model(x, y, invvar, modelxy, hyperpars):
    return np.argmax(ln_likelihood(x, y, invvar, modelxy))

def logsum(x):
    offset = 700 - np.max(x) - np.log(len(x))
    return np.log(np.sum(np.exp(x + offset))) - offset

def ln_prior(hyperpars):
    return hyperpars - logsum(hyperpars)

def ln_posterior(x, y, invvar, modelxy, hyperpars):
    return ln_prior(hyperpars) + ln_likelihood(x, y, invvar, modelxy)

def marginalized_ln_likelihood(x, y, invvar, modelxy, hyperpars, modelclass, c):
    I = (modelclass == c)
    return logsum(ln_posterior(x, y, invvar, modelxy, hyperpars)[I])

def total_marginalized_ln_likelihood(xy, invvar, modelxy, hyperpars, modelclass, c):
    x, y = xy
    mll = [marginalized_ln_likelihood(x[i], y[i], invvar[i], modelxy, hyperpars, modelclass, c) for i in range(len(x))]
    return np.sum(mll)

def objective(hyperpars, xy, invvar, modelxy, modelclass, c):
    return -1. * total_marginalized_ln_likelihood(xy, invvar, modelxy, hyperpars, modelclass, c)

def marginalized_blind_ln_likelihood(x, y, invvar, modelxy, hyperpars):
    return logsum(ln_posterior(x, y, invvar, modelxy, hyperpars))

def total_marginalized_blind_ln_likelihood(xy, invvar, modelxy, hyperpars):
    x, y = xy
    mll = [marginalized_blind_ln_likelihood(x[i], y[i], invvar[i], modelxy, hyperpars) for i in range(len(x))]
    return np.sum(mll)

def objective_blind(hyperpars, xy, invvar, modelxy):
    return -1. * total_marginalized_blind_ln_likelihood(xy, invvar, modelxy, hyperpars)

def plot_internal(x, y, c, label):
    marker = 'ko'
    a = 0.5
    for i in [1, 0]:
        if label and i == 0:
            marker = 'go'
            a = 0.5
        if label and i == 1:
            marker = 'bo'
            a = 0.5
        I = (c == i)
        if np.sum(I) > 0:
            plt.plot(x[I], y[I], marker, mew=0, alpha=a)
    return None

def plot_class(xy, c, fn, title='', modelxy=None, modelc=None, hpars=None, label=True):
    x, y = xy
    plt.clf()
    plot_internal(x, y, c, label)
    if modelxy is not None:
        mx, my = modelxy
        alphasum = [logsum(hpars[modelc == i]) for i in range(2)]
        for i in range(len(mx)):
            if modelc[i] == 0:
                marker = 'r+'
                mew = 15. * np.exp(0.5 * (hpars[i] - alphasum[0]))
                ms = 45. * np.exp(0.5 * (hpars[i] - alphasum[0]))
            else:
                alpha = np.exp(hpars[i] - alphasum[1])
                marker = 'rx'
                mew = 15. * np.exp(0.5 * (hpars[i] - alphasum[1]))
                ms = 35. * np.exp(0.5 * (hpars[i] - alphasum[1]))
            plt.plot([mx[i], ], [my[i], ], marker, alpha=0.75, mew=mew, ms=ms)
    plt.xlim(-2., 5.)
    plt.ylim(-1.5, 4.5)
    plt.title(title)
    plt.savefig(fn)
    print 'plot_class: wrote ' + fn
    return None

def plot_two_classes(xy, truec, plotc, prefix, method):
    stars = (plotc == 0)
    starxy = (qq[stars] for qq in xy)
    contam = 100. * sum(truec[stars]) / sum(1-truec[stars])
    title = '%s / stars / contamination %.2f percent' % (method, contam)
    plot_class(starxy, truec[stars], prefix + '-stars.png', title=title)
    gals = (plotc == 1)
    galxy = (qq[gals] for qq in xy)
    contam = 100. * sum(1-truec[gals]) / sum(truec[gals])
    title = '%s / galaxies / contamination %.2f percent' % (method, contam)
    plot_class(galxy, truec[gals], prefix + '-gals.png', title=title)
    return None

def main():
    np.random.seed(42)
    truexy, trueclass = make_truth()
    plot_class(truexy, trueclass, 'toy-truth.png')
    noisyxy, noisyclass, noisyinvvar = noisify(truexy, trueclass)
    plot_class(noisyxy, trueclass, 'toy-noisy.png')
    plot_class(noisyxy, trueclass, 'toy-nolab.png', label=False)
    modelxy, modelclass = make_models()
    hyperpars = np.ones(len(modelclass)).astype('float')
    plot_class(truexy, trueclass, 'toy-models-truth.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
    plot_class(noisyxy, trueclass, 'toy-models-noisy.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
    plot_class(noisyxy, trueclass, 'toy-models-nolab.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars, label=False)
    plot_two_classes(noisyxy, trueclass, trueclass, 'toy-true', 'truth')
    x, y = noisyxy
    maxlclass = np.zeros(len(x)) - 1
    for i in range(len(x)):
        maxlclass[i] = modelclass[ml_model(x[i], y[i], noisyinvvar[i], modelxy, hyperpars)]
    plot_two_classes(noisyxy, trueclass, maxlclass, 'toy-maxl', 'maximum likelihood')
    marginalizedclass = np.zeros(len(x))
    for i in range(len(x)):
        if marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 1) > marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 0):
            marginalizedclass[i] = 1
    plot_two_classes(noisyxy, trueclass, marginalizedclass, 'toy-flat', 'flat priors')

    # optimize WITHOUT using trueclass
    for maxfit in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        ndata = len(noisyinvvar)
        J = np.random.permutation(ndata)
        if len(J) > maxfit:
            J = J[:maxfit]
        thatx, thaty = noisyxy
        thisnoisyxy = (thatx[J], thaty[J])
        thisinvvar = noisyinvvar[J]
        args = (thisnoisyxy, thisinvvar, modelxy)
        besthyperpars = op.fmin(objective_blind, hyperpars, args=args, maxiter=10000)
        thishyperpars = besthyperpars - logsum(besthyperpars)
        hyperpars = thishyperpars
        print hyperpars
        plot_class(truexy, trueclass, 'toy-models-hier-blind-truth.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
        plot_class(noisyxy, trueclass, 'toy-models-hier-blind-noisy.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
        plot_class(noisyxy, trueclass, 'toy-models-hier-blind-nolab.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars, label=False)
        for i in range(len(x)):
            if marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 1) > marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 0):
                marginalizedclass[i] = 1
        plot_two_classes(noisyxy, trueclass, marginalizedclass, 'toy-hier-blind', 'hierarchical')

    # split into star and galaxy models for optimization; optimize
    for maxfit in [30, 40, 50, 60, 80, 100, 200]:
        for c in [0, 1]:
            I = np.flatnonzero(modelclass == c)
            thisx, thisy = modelxy
            thismodelxy = (thisx[I], thisy[I])
            thismodelclass = modelclass[I]
            assert(np.sum(thismodelclass == (1-c)) == 0)
            thishyperpars = hyperpars[I]
            J = np.random.permutation(np.flatnonzero(trueclass == c))
            if len(J) > maxfit:
                J = J[:maxfit]
            thatx, thaty = noisyxy
            thisnoisyxy = (thatx[J], thaty[J])
            thisinvvar = noisyinvvar[J]
            args = (thisnoisyxy, thisinvvar, thismodelxy, thismodelclass, c)
            besthyperpars = op.fmin(objective, thishyperpars, args=args, maxiter=10000)
            thishyperpars = besthyperpars - logsum(besthyperpars)
            print thishyperpars
            hyperpars[I] = thishyperpars - logsum(thishyperpars) + np.log(len(I))
        print hyperpars
        plot_class(truexy, trueclass, 'toy-models-hier-train-truth.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
        plot_class(noisyxy, trueclass, 'toy-models-hier-train-noisy.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
        plot_class(noisyxy, trueclass, 'toy-models-hier-train-nolab.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars, label=False)
        for i in range(len(x)):
            if marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 1) > marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 0):
                marginalizedclass[i] = 1
        plot_two_classes(noisyxy, trueclass, marginalizedclass, 'toy-hier-train', 'hierarchical w training')

    return None

if __name__ == '__main__':
    main()
