import math as ma
import matplotlib
matplotlib.use('Agg')
# pylab must go before numpy.random import
from pylab import *
from numpy import *
import numpy.random as random
from generate_data import read_data
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# a colormap that goes from white to black: the opposite of matplotlib.gray()
antigray = LinearSegmentedColormap('antigray',
								   {'red':   ((0., 1, 1), (1., 0, 0)),
									'green': ((0., 1, 1), (1., 0, 0)),
									'blue':  ((0., 1, 1), (1., 0, 0))})

rcParams.update({
	'axes.labelsize': 12,
	'text.fontsize': 11,
	'text.usetex': True,
	'legend.fontsize': 12,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'figure.figsize': [5,5],
	'image.interpolation':'nearest',
	'image.origin':'lower',
	})

# returns an array of the single-point likelihoods.
def single_point_likelihoods(x, y, yvar, m, b, Pbad, Ybad, Vbad):
	return ((1 - Pbad) / sqrt(2.*pi*yvar) * exp(-0.5 * (y - m*x - b)**2 / yvar) +
			Pbad / sqrt(2.*pi * Vbad) * exp(-0.5 * (y - Ybad)**2 / Vbad))

def likelihood(x, y, yvar, m, b, Pbad, Ybad, Vbad):
	return prod(single_point_likelihoods(x, y, yvar, m, b, Pbad, Ybad, Vbad))

# Technically an improper prior (we just do naive range checks).  Feel free to insert your own.
def prior(m, b, Pbad, Ybad, Vbad):
	return (Pbad >= 0) * (Pbad < 1) * (Vbad > 0)

# Not properly normalized (with the original prior)
def posterior(x, y, yvar, m, b, Pbad, Ybad, Vbad):
	return likelihood(x, y, yvar, m, b, Pbad, Ybad, Vbad) * prior(m, b, Pbad, Ybad, Vbad)

def pick_new_parameters(nsteps, m, b, Pbad, Ybad, Vbad):
	mscale = 0.05
	bscale = 1.
	# burn-in slope and intercept
	if nsteps > 10000:
		pbadscale = 0.1
		ybadscale = bscale
		vbadscale = 10.
	else:
		pbadscale = 0
		ybadscale = 0
		vbadscale = 0
	return (m + mscale * random.normal(),
			b + bscale * random.normal(),
			Pbad + pbadscale * random.normal(),
			Ybad + ybadscale * random.normal(),
			Vbad + vbadscale * random.normal())

def marginalize_mixture(mixture=True, thirds=False, short=False):
	if mixture:
		prefix = 'mixture'
	else:
		prefix = 'nomixture'

	if thirds:
		prefix += '-thirds'

	random.seed(-1) #In the interest of reproducibility (if that's a word)
	#Read the data
	data= read_data('data_yerr.dat')
	ndata= len(data)
	#Put the data in the appropriate arrays and matrices
	x= zeros(ndata)
	y= zeros(ndata)
	yvar= zeros(ndata)
	jj= 0
	for ii in arange(ndata):
		x[jj]= data[ii][1][0]
		y[jj]= data[ii][1][1]
		yvar[jj]= data[ii][2]**2
		jj= jj+1
	ndata= jj
	x= x[0:ndata]
	y= y[0:ndata]
	yvar= yvar[0:ndata]

	if thirds:
		yvar /= 9.

	# initialize parameters
	m = (y[7]-y[9]) / (x[7]-x[9])
	b = y[7] - m * x[7]
	if mixture:
		Pbad = 0.5
	else:
		Pbad = 0.
	Ybad = mean(y)
	Vbad = mean((y-Ybad)**2)

	p = posterior(x, y, yvar, m, b, Pbad, Ybad, Vbad)
	print 'starting p=', p

	chain = []
	oldp = p
	oldparams = (m, b, Pbad, Ybad, Vbad)
	bestparams = oldparams
	bestp = oldp

	nsteps = 0
	naccepts = 0

	NSTEPS = 100000
	if short:
		NSTEPS /= 2
	print 'doing', NSTEPS, 'steps of MCMC...'
	while nsteps < NSTEPS:
		newparams = pick_new_parameters(nsteps, *oldparams)
		if not mixture:
			# clamp Pbad to zero.
			(m, b, Pbad, Ybad, Vbad) = newparams
			newparams = (m, b, 0, Ybad, Vbad)

		p = posterior(x, y, yvar, *newparams)
		if p/oldp > random.uniform():
			chain.append((p,newparams))
			oldparams = newparams
			oldp = p
			if p > bestp:
				bestp = p
				bestparams = newparams
			naccepts += 1
		else:
			chain.append((oldp,oldparams))
			# keep oldparams, oldp
		nsteps += 1
		if (nsteps % 5000 == 1):
			print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams

	print 'acceptance fraction', (naccepts/float(nsteps))

	# plot a sample
	

	# Plot data
	errorbar(x, y, sqrt(yvar), color='k', marker='o', linestyle='None')
	xlabel(r'$x$')
	ylabel(r'$y$')
	xlim(0,300)
	ylim(0,700)
	savefig(prefix + '-data.pdf')

	a = axis()
	xmin, xmax = xlim()
	ymin, ymax = ylim()
	xs = linspace(xmin, xmax, 2)
	# select 10 samples at random from the second half of the chain.
	Nchain = len(chain)
	if mixture:
		I = Nchain/2 + random.permutation(Nchain/2)[:10]
	else:
		I = array([argmax([p for (p, params) in chain])])
	for i in I:
		(p,params) = chain[i]
		(m, b, Pbad, Ybad, Vbad) = params
		ys = m * xs + b
		plot(xs, ys, color='k', alpha=0.3)
	axis(a)
	savefig(prefix + '-xy.pdf')

	if mixture:
		bgp = zeros(len(x))
		fgp = zeros(len(x))
		for (p,params) in chain[Nchain/2:]:
			(m, b, Pbad, Ybad, Vbad) = params
			bgp += Pbad      * single_point_likelihoods(x, y, yvar, m, b, 1, Ybad, Vbad)
			fgp += (1.-Pbad) * single_point_likelihoods(x, y, yvar, m, b, 0, Ybad, Vbad)
		bgodds = bgp / fgp
		for i,bgo in enumerate(bgodds):
			if bgo < 1:
				continue
			dxl = (xmax-xmin) * 0.01
			dyl = (ymax-ymin) * 0.01
			t = text(x[i]+dxl, y[i]+dyl, '%.1f' % log10(bgo),
					 horizontalalignment='left',
					 verticalalignment='bottom', alpha=0.3)
		savefig(prefix + '-xy-bg.pdf')

	clf()
	ms = array([m for (p, (m, b, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
	bs = array([b for (p, (m, b, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
	#plot(ms, bs, 'k,', alpha=0.1)
	#xlabel('slope $m$')
	#ylabel('intercept $b$')
	#savefig(prefix + '-mb-scatter.pdf')

	clf()
	(H, xe, ye) = histogram2d(ms, bs, bins=(100,100))
	print 'max H:', H.max()
	imshow(log(1 + H.T), extent=(xe.min(), xe.max(), ye.min(), ye.max()), aspect='auto',
		   cmap=antigray)
	xlabel('slope $m$')
	ylabel('intercept $b$')
	savefig(prefix + '-mb.pdf')

if __name__ == '__main__':
	args = sys.argv[1:]
	if 'black' in args:
		print 'making black page, you tool'
		axes([0,0,1,1], axisbg='k')
		savefig('black.pdf')
		sys.exit(0)
	thirds = 'thirds' in args
	quick = 'quick' in args
	mixture = not('nomixture' in args)
	marginalize_mixture(mixture, thirds, quick)
