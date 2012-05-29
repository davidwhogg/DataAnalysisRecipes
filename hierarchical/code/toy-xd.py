"""
This file is part of Data Analysis Recipes.
Copyright 2012 David W. Hogg (NYU).
"""

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
    rc('text', usetex=True)
import numpy as np
import pylab as plt

class gauss_mix_1d():

    def __init__(self, amp, mean, var):
        self.amp = np.array(amp)
        self.mean = np.array(mean)
        self.var = np.array(var)
        self.K = self.amp.size
        assert self.amp.shape == (self.amp.size, )
        assert self.mean.shape == self.amp.shape
        assert self.var.shape == self.amp.shape
        return None

    def __getitem__(self, k):
        return self.amp[k], self.mean[k], self.var[k]

    def evaluate(self, x):
        y = np.zeros_like(x).astype(float)
        for a,m,v in self:
            y += a * np.exp(-0.5 * (x - m)**2 / v) / np.sqrt(2. * np.pi * v)
        return y

    def get_one_sample(self):
        q1 = np.cumsum(self.amp)
        q1 /= q1[-1]
        q2 = np.append(0., q1[0:-1])
        r = np.random.uniform()
        I = (r < q1) * (r > q2)
        k = (np.arange(self.K)[I])[0]
        print k
        a, m, v = self[k]
        return m + np.sqrt(v) * np.random.normal()

    def convolve(self, var):
        return gauss_mix_1d(self.amp, self.mean, self.var + var)

if __name__ == "__main__":
    np.random.seed(42)
    dpi = 200
    truth = gauss_mix_1d([0.5, 0.4, 0.1], [-0.1, 0.2, 0.25], [0.005, 0.02, 0.1])
    nsample = 200
    obsvars = 0.02 * np.random.uniform(size=nsample)
    xs = np.zeros_like(obsvars)
    xlim = [-1, 1]
    xp = np.arange(-1., 1., 0.001)
    truealpha = 0.5
    plt.figure(figsize=(5,3.5))
    for n,obsvar in enumerate(obsvars):
        observed = truth.convolve(obsvar)
        xs[n] = observed.get_one_sample()
        if n < 10:
            plt.clf()
            plt.plot(xp, truth.evaluate(xp), 'k-', alpha=truealpha)
            plt.plot(xp, observed.evaluate(xp), 'k-', alpha=1.)
            y0, y1 = plt.ylim()
            ys = 0.5 * (y1 + y0)
            plt.plot(xs[n], ys, 'ko')
            plt.plot([xs[n] - np.sqrt(obsvar), xs[n] + np.sqrt(obsvar)], [ys, ys], 'k-', alpha=1.)
            plt.xlim(xlim)
            plt.xlabel('$x$')
            plt.savefig('toy-xd-%02d.png' % n, dpi=dpi)
    plt.clf()
    sampleIDs = np.arange(nsample) + 1.
    plt.plot(xs, sampleIDs, 'ko')
    for x, y, obsvar in zip(xs, sampleIDs, obsvars):
        plt.plot([x - np.sqrt(obsvar), x + np.sqrt(obsvar)], [y, y], 'k-', alpha=1.)
    plt.xlim(xlim)
    plt.xlabel('$x$')
    plt.ylim(-0.5, nsample + 1.5)
    plt.ylabel('sample ID')
    plt.savefig('toy-xd-sample.png', dpi=dpi)
    plt.clf()
    plt.plot(xp, truth.evaluate(xp), 'k-', lw=1, alpha=truealpha)
    plt.xlim(xlim)
    plt.xlabel('$x$')
    plt.savefig('toy-xd-truth.png', dpi=dpi)
    nbins = 30
    plt.hist(xs, color='k', bins=nbins, normed=True, histtype='step')
    plt.xlim(xlim)
    plt.savefig('toy-xd-hist.png', dpi=dpi)
