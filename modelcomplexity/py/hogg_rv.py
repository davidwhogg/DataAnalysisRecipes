'''
This file is part of the Data Analysis Recipes project.
Copyright 2011 David W. Hogg (NYU)
'''

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import scipy.optimize as op
import pylab as plt

plotsuffix = '.png'
halflnpi = 0.5 * np.log(np.pi)

def read_data(fn):
    f = open(fn, 'r')
    data = []
    for line in f:
        if line[0] == '#':
            continue
        data.append(np.array([float(word) for word in line.split()]))
    data = np.array(data)
    time = data[:,0]
    rv = data[:,1]
    invvar = 1.e6 / data[:,2]**2
    info = (invvar, time)
    return rv, info

# rv: numpy array of N data points
# invvar: either 1 or N inverse variance values
# modelrv: either 1 or N model rv values
# jitter: either 1 or N noise amplitudes to be added in quadrature to the input noise
def lnlikelihood_Gaussian(rv, invvar, modelrv, jitter=0.):
    newinvvar = 1. / (jitter**2 + 1. / invvar)
    return 0.5 * np.sum(np.log(newinvvar)) - halflnpi - 0.5 * np.sum(newinvvar * (rv - modelrv)**2)

def rv_sinusoid(time, A, B, omega):
    return A * np.cos(omega * time) + B * np.sin(omega * time)

def lnlikelihood_sinusoid_Gaussian(rv, invvar, time, modelrv, A, B, omega, jitter=0., trend=0.):
    return lnlikelihood_Gaussian(rv, invvar, modelrv + trend * (time - 2450000.) + rv_sinusoid(time, A, B, omega), jitter)

def lnlikelihood(pars, data, model, info):
    invvar, time = info
    if model == 'constant':
        return lnlikelihood_Gaussian(data, invvar, pars[0])
    if model == 'jitter':
        return lnlikelihood_Gaussian(data, invvar, pars[0], pars[1])
    if model == 'sinusoid':
        return lnlikelihood_sinusoid_Gaussian(data, invvar, time, pars[0], pars[1], pars[2], pars[3])
    if model == 'sinusoid+jitter':
        return lnlikelihood_sinusoid_Gaussian(data, invvar, time, pars[0], pars[1], pars[2], pars[3], pars[4])
    if model == 'sinusoid+jitter+trend':
        return lnlikelihood_sinusoid_Gaussian(data, invvar, time, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5])

def cost(pars, data, model, info):
    return -1. * lnlikelihood(pars, data, model, info)

def optimize_likelihood(pars, data, model, info):
    bestpars = op.fmin(cost, pars, args=(data, model, info))
    return bestpars

def hogg_errorbar(x, y, invvar, color, alpha=1.):
    for x, y, yerr in zip(x, y, 1. / np.sqrt(invvar)):
        plt.plot([x, x], [y-yerr, y+yerr], color, alpha=alpha)
    return

def make_plot(pars, data, model, info, fn):
    plt.clf()
    invvar, time = info
    t1 = int(np.median(time))
    dtime = time - t1
    rv1 = int(np.median(data))
    drv = data - rv1
    plt.plot(dtime, drv, 'k.', alpha=0.5)
    hogg_errorbar(dtime, drv, invvar, 'k', alpha=0.5)
    timegrid = np.arange(np.min(time)-30., np.max(time)+30., 0.1)
    if model == 'constant':
        modelrv = 0. * timegrid + pars[0]
        plt.plot(timegrid - t1, modelrv - rv1, 'k-', alpha=0.5)
    if model == 'jitter':
        modelrv = 0. * timegrid + pars[0]
        modelrvlo = modelrv - pars[1]
        modelrvhi = modelrv + pars[1]
        plt.plot(timegrid - t1, modelrvlo - rv1, 'k-', alpha=0.5)
        plt.plot(timegrid - t1, modelrvhi - rv1, 'k-', alpha=0.5)
    if model == 'sinusoid':
        modelrv = pars[0] + rv_sinusoid(timegrid, pars[1], pars[2], pars[3])
        plt.plot(timegrid - t1, modelrv - rv1, 'k-', alpha=0.5)
    if model == 'sinusoid+jitter':
        modelrv = pars[0] + rv_sinusoid(timegrid, pars[1], pars[2], pars[3])
        modelrvlo = modelrv - pars[4]
        modelrvhi = modelrv + pars[4]
        plt.plot(timegrid - t1, modelrvlo - rv1, 'k-', alpha=0.5)
        plt.plot(timegrid - t1, modelrvhi - rv1, 'k-', alpha=0.5)
    if model == 'sinusoid+jitter+trend':
        modelrv = pars[0] + pars[5] * (timegrid - 2450000.) + rv_sinusoid(timegrid, pars[1], pars[2], pars[3])
        modelrvlo = modelrv - pars[4]
        modelrvhi = modelrv + pars[4]
        plt.plot(timegrid - t1, modelrvlo - rv1, 'k-', alpha=0.5)
        plt.plot(timegrid - t1, modelrvhi - rv1, 'k-', alpha=0.5)
    plt.xlim(np.min(timegrid - t1), np.max(timegrid - t1))
    plt.xlabel('time relative to %d (d)' % t1)
    plt.ylabel('radial velocity relative to %d (m/s)' % rv1)
    plt.title('model: %s. ln likelihood: %f' % (model, lnlikelihood(pars, data, model, info)))
    plt.savefig(fn)
    return

def scan_sinusoid(data, info):
    model = 'sinusoid'
    lowestcost = np.Inf
    for omega in np.arange(0.001, 1., 0.002):
        p = np.array([np.median(data), 0.01, 0.01, omega])
        p = optimize_likelihood(p, data, model, info)
        c = cost(p, data, model, info)
        if c < lowestcost:
            print p, c
            lowestcost = c
            bestpars = p
    return bestpars

def fit_and_plot(pars, data, model, info):
    bestpars = optimize_likelihood(pars, data, model, info)
    print 'before', pars, 'after', bestpars
    make_plot(bestpars, data, model, info, model + plotsuffix)
    return bestpars

def old_main():
    data, info = read_data('HD104067.dat')
    model = 'constant'
    pars = np.array([np.median(data)])
    pars = fit_and_plot(pars, data, model, info)
    model = 'jitter'
    pars = np.array([np.median(data), 0.1])
    pars = fit_and_plot(pars, data, model, info)
    model = 'sinusoid'
    # pars found with scan_sinusoid!
    pars = np.array([1.51226917e+01, 8.87236709e-03, 8.84415337e-03, 1.12435113e-01])
    pars = fit_and_plot(pars, data, model, info)
    model = 'sinusoid+jitter'
    pars = np.append(pars, [0.01])
    fit_and_plot(pars, data, model, info)
    model = 'sinusoid+jitter+trend'
    pars = np.append(pars, [-1.e-9])
    fit_and_plot(pars, data, model, info)
    return pars, data, model, info

def loo_lnlikelihood(pars, data, model, info, index=None):
    Ndata = len(data)
    invvar, time = info
    use = (np.arange(Ndata) != index)
    shortpars = optimize_likelihood(pars, data[use], model, (invvar[use], time[use]))
    return lnlikelihood(shortpars, data[index], model, (invvar[index], time[index]))

def crossvalidate_lnlikelihood(pars, data, model, info):
    return np.sum(np.array([loo_lnlikelihood(pars, data, model, info, index) for index in range(len(data))]))

def main():
    data, info = read_data('HD104067.dat')
    model = 'sinusoid'
    pars = np.array([1.51226917e+01, 8.87236709e-03, 8.84415337e-03, 1.12435113e-01])
    pars = optimize_likelihood(pars, data, model, info)
    model = 'sinusoid+jitter'
    pars = np.append(pars, [0.01])
    pars = optimize_likelihood(pars, data, model, info)
    sjll = crossvalidate_lnlikelihood(pars, data, model, info)
    model = 'sinusoid+jitter+trend'
    pars = np.append(pars, [1.e-9])
    pars = optimize_likelihood(pars, data, model, info)
    sjtll = crossvalidate_lnlikelihood(pars, data, model, info)
    print sjll, sjtll

if __name__ == "__main__":
    main()

if False:
    os.system("yes | rm -rfi /")
