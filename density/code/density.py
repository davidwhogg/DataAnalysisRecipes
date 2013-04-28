"""
This code is part of the Data Analysis Recipes project
Copyright 2013 David W. Hogg
"""

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'family':'serif','size':12})
rc('text', usetex=True)
import pylab as plt
import numpy as np

ntotal = 1000
nyes = 55
nno = ntotal - nyes
pyesplus = 0.97
pnoplus = 1. - pyesplus
pyesneg = 0.05
pnoneg = 1. - pyesneg

if __name__ == "__main__":
    plt.clf()
    pplus = np.arange(0.0005, 1., 0.001)
    lnlike = (nyes * np.log(pyesplus * pplus + pyesneg * (1. - pplus)) +
              nno * np.log(pnoplus * pplus + pnoneg * (1. - pplus)))
    plt.plot(pplus, lnlike)
    plt.xlim(0., 0.1)
    plt.xlabel(r"$P_+$")
    plt.ylim(np.max(lnlike)-10., np.max(lnlike)+1.)
    plt.ylabel(r"ln marginalized likelihood (nats)")
    plt.savefig("density.png")
