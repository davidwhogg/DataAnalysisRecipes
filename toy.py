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
    nstar = 70
    ngalaxy = 300
    ystar = np.random.uniform(-0.5, 3.5, size=nstar)
    xstar = 0.1 * ystar**2 + 0.02 * np.random.normal(size=nstar)
    classstar = np.zeros(nstar).astype(int)
    xgalaxy = np.random.uniform(-0.5, 3.5, size=ngalaxy)
    ygalaxy = 0.1 * (xgalaxy - 0.5)**2 + 0.02 * np.random.normal(size=ngalaxy)
    classgalaxy = np.ones(ngalaxy).astype(int)
    x = np.append(xstar, xgalaxy)
    y = np.append(ystar, ygalaxy)
    c = np.append(classstar, classgalaxy)
    return ((x, y), c)

def make_models():
    x = np.array([])
    y = np.array([])
    c = np.array([])
    tx, ty = np.meshgrid(np.arange(0.0, 0.51, 0.2), np.arange(-0.5, 2.01, 0.2))
    tx = tx.ravel()
    ty = ty.ravel()
    tc = np.zeros(len(tx)).astype(int)
    x = np.append(x, tx)
    y = np.append(y, ty)
    c = np.append(c, tc)
    tx, ty = np.meshgrid(np.arange(-0.5, 3.01, 0.3), np.arange(0., 0.61, 0.3))
    tx = tx.ravel()
    ty = ty.ravel()
    tc = np.ones(len(tx)).astype(int)
    x = np.append(x, tx)
    y = np.append(y, ty)
    c = np.append(c, tc)
    return ((x, y), c)

# serious duck typing in this function
def ln_likelihood(x, y, invvar, modelxy):
    mx, my = modelxy
    return -0.5 * invvar * ((x - mx)**2 + (y - my)**2)

def map_model(x, y, invvar, modelxy, hyperpars):
    return np.argmax(ln_posterior(x, y, invvar, modelxy, hyperpars))

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
    nx = len(x)
    mll = np.zeros(nx).astype('float')
    for i in range(nx):
        mll[i] = marginalized_ln_likelihood(x[i], y[i], invvar[i], modelxy, hyperpars, modelclass, c)
    return logsum(mll)

def objective(hyperpars, xy, invvar, modelxy, modelclass, c):
    return -1. * total_marginalized_ln_likelihood(xy, invvar, modelxy, hyperpars, modelclass, c)

def noisify(xy, c):
    nobj = len(c)
    x,y = xy
    invvar = 1.0 / (0.05 + 0.35 * np.random.uniform(size=nobj))**2
    nx = x + np.random.normal(size=nobj)/np.sqrt(invvar)
    ny = y + np.random.normal(size=nobj)/np.sqrt(invvar)
    bad = (np.random.uniform(size=nobj) < 0.01)
    nc = np.zeros(nobj)
    nc[:] = c[:]
    nc[bad] = 1 - nc[bad]
    return ((nx, ny), nc, invvar)

def plot_internal(x, y, c, star, galaxy, alpha):
    for i in [1, 0]:
        if i == 0:
            marker = star
        else:
            marker = galaxy
        I = (c == i)
        if np.sum(I) > 0:
            plt.plot(x[I], y[I], marker, alpha=alpha)
    return None

def plot_class(xy, c, fn, modelxy=None, modelc=None, hpars=None):
    x, y = xy
    plt.clf()
    plot_internal(x, y, c, 'ko', 'bo', 0.25)
    if modelxy is not None:
        mx, my = modelxy
        alphas = np.maximum(0., (hpars + 60. - np.max(hpars)) / 60.)
        for i in range(len(mx)):
            if modelc[i] == 0:
                marker = 'r+'
            else:
                marker = 'rx'
            plt.plot([mx[i], ], [my[i], ], marker, alpha=alphas[i])
    plt.xlim(-2., 5.)
    plt.ylim(-1.5, 4.5)
    plt.savefig(fn)
    print 'plot_class: wrote ' + fn
    return None

def plot_two_classes(xy, truec, plotc, prefix):
    stars = (plotc == 0)
    starxy = (qq[stars] for qq in xy)
    plot_class(starxy, truec[stars], prefix + '-stars.png')
    gals = (plotc == 1)
    galxy = (qq[gals] for qq in xy)
    plot_class(galxy, truec[gals], prefix + '-gals.png')
    return None

def main():
    np.random.seed(42)
    truexy, trueclass = make_truth()
    plot_class(truexy, trueclass, 'toy-truth.png')
    noisyxy, noisyclass, noisyinvvar = noisify(truexy, trueclass)
    plot_class(noisyxy, trueclass, 'toy-noisy.png')
    modelxy, modelclass = make_models()
    hyperpars = np.ones(len(modelclass)).astype('float')
    plot_class(truexy, trueclass, 'toy-models-truth.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
    plot_class(noisyxy, trueclass, 'toy-models-noisy.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
    plot_two_classes(noisyxy, trueclass, trueclass, 'toy-true')
    x, y = noisyxy
    maxlclass = np.zeros(len(x)) - 1
    for i in range(len(x)):
        maxlclass[i] = modelclass[map_model(x[i], y[i], noisyinvvar[i], modelxy, hyperpars)]
    plot_two_classes(noisyxy, trueclass, maxlclass, 'toy-maxl')
    marginalizedclass = np.zeros(len(x))
    for i in range(len(x)):
        if marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 1) > marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 0):
            marginalizedclass[i] = 1
    plot_two_classes(noisyxy, trueclass, marginalizedclass, 'toy-marginal')

    # split into star and galaxy models for optimization; optimize
    for c in [0, 1]:
        I = (modelclass == c)
        thisx, thisy = modelxy
        thismodelxy = (thisx[I], thisy[I])
        thismodelclass = modelclass[I]
        thishyperpars = hyperpars[I]
        J = (trueclass == c)
        thatx, thaty = noisyxy
        thisnoisyxy = (thatx[J], thaty[J])
        thisinvvar = noisyinvvar[J]
        args = (thisnoisyxy, thisinvvar, thismodelxy, thismodelclass, c)
        besthyperpars = op.fmin(objective, thishyperpars, args=args, maxiter=300)
        thishyperpars = besthyperpars - logsum(besthyperpars)
        print thishyperpars
        besthyperpars = op.fmin(objective, thishyperpars, args=args, maxiter=300)
        thishyperpars = besthyperpars - logsum(besthyperpars)
        print thishyperpars
        besthyperpars = op.fmin(objective, thishyperpars, args=args, maxiter=300)
        thishyperpars = besthyperpars - logsum(besthyperpars)
        print thishyperpars
        hyperpars[I] = thishyperpars - logsum(thishyperpars) + np.log(len(thishyperpars)) - np.log(len(hyperpars))
    print hyperpars

    plot_class(truexy, trueclass, 'toy-models-hier-truth.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
    plot_class(noisyxy, trueclass, 'toy-models-hier-noisy.png', modelxy=modelxy, modelc=modelclass, hpars=hyperpars)
    for i in range(len(x)):
        if marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 1) > marginalized_ln_likelihood(x[i], y[i], noisyinvvar[i], modelxy, hyperpars, modelclass, 0):
            marginalizedclass[i] = 1
    plot_two_classes(noisyxy, trueclass, marginalizedclass, 'toy-marginal-hier')

    return None

if __name__ == '__main__':
    main()
