# This code is filled with expressions that blow up at theta=0.5*pi;
# these can be replaced with much better expressions that make use of
# linear algebra; this is left as an exercise to the reader!

import math as ma
import matplotlib
matplotlib.use('Agg')
# pylab must go before numpy.random import
from pylab import *
from numpy import *
import numpy.random as random
from generate_data import read_data
from matplotlib import rcParams
import scipy.linalg as linalg
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap

# a colormap that goes from white to black: the opposite of matplotlib.gray()
antigray = LinearSegmentedColormap('antigray',
								   {'red':   ((0., 1, 1), (1., 0, 0)),
									'green': ((0., 1, 1), (1., 0, 0)),
									'blue':  ((0., 1, 1), (1., 0, 0))})

# returns an array of the single-point likelihoods.
# some good linear algebra would speed this up a lot
def single_point_likelihoods(x, y, yvar, theta, bperp, Pbad, Ybad, Vbad):
	projvar = zeros(len(x))
	projpos = zeros(len(x))
	unitv = array([-sin(theta),cos(theta)])
	for i,(xi,yi,yvari) in enumerate(zip(x,y,yvar)):
		projvar[i] = dot(unitv, dot(yvari, unitv.T))
		projpos[i] = dot(unitv, array([xi,yi]).T)
	return ((1 - Pbad) / sqrt(2.*pi*projvar) * exp(-0.5 * (projpos - bperp)**2 / projvar) +
			Pbad / sqrt(2.*pi * Vbad) * exp(-0.5 * (projpos - Ybad)**2 / Vbad))

def likelihood(x, y, yvar, theta, bperp, Pbad, Ybad, Vbad):
	return prod(single_point_likelihoods(x, y, yvar, theta, bperp, Pbad, Ybad, Vbad))

# Technically an improper prior (we just do naive range checks).  Feel free to insert your own.
def prior(theta, bperp, Pbad, Ybad, Vbad):
	return (Pbad >= 0) * (Pbad < 1) * (Vbad > 0)

# Not properly normalized (with the original prior)
def posterior(x, y, yvar, theta, bperp, Pbad, Ybad, Vbad):
	return likelihood(x, y, yvar, theta, bperp, Pbad, Ybad, Vbad) * prior(theta, bperp, Pbad, Ybad, Vbad)

def pick_new_parameters(nsteps, theta, bperp, Pbad, Ybad, Vbad):
	thetascale = 0.01
	bperpscale = 1.
	# burn-in slope and intercept
	if nsteps > 10000:
		pbadscale = 0.1
		ybadscale = bperpscale
		vbadscale = 10.
	else:
		pbadscale = 0
		ybadscale = 0
		vbadscale = 0
	return (theta + thetascale * random.normal(),
			bperp + bperpscale * random.normal(),
			Pbad + pbadscale * random.normal(),
			Ybad + ybadscale * random.normal(),
			Vbad + vbadscale * random.normal())

def marginalize_mixture(mixture=True, short=False):
	if mixture:
		prefix = 'mixture2d'
	else:
		prefix = 'nomixture2d'

	random.seed(-1) #In the interest of reproducibility (if that's a word)
	# Read the data
	data= read_data('data_allerr.dat',True)
	ndata= len(data)
	# Create the ellipses and the data points
	x= zeros(ndata)
	y= zeros(ndata)
	ellipses=[]
	yvar= zeros((ndata,2,2))
	for ii in range(ndata):
		x[ii]= data[ii][1][0]
		y[ii]= data[ii][1][1]
		#Calculate the eigenvalues and the rotation angle
		yvar[ii,0,0]= data[ii][3]**2.
		yvar[ii,1,1]= data[ii][2]**2.
		yvar[ii,0,1]= data[ii][4]*sqrt(yvar[ii,0,0]*yvar[ii,1,1])
		yvar[ii,1,0]= yvar[ii,0,1]
		eigs= linalg.eig(yvar[ii,:,:])
		angle= arctan(-eigs[1][0,1]/eigs[1][1,1])/pi*180.
		thisellipse= Ellipse(array([x[ii],y[ii]]),2*sqrt(eigs[0][0]),
							 2*sqrt(eigs[0][1]),angle)
		ellipses.append(thisellipse)

	# initialize parameters
	theta = arctan2(y[7]-y[9],x[7]-x[9])
	bperp = (y[7] - tan(theta) * x[7]) * cos(theta) # bad at theta = 0.5 * pi
	if mixture:
		Pbad = 0.5
	else:
		Pbad = 0.
	Ybad = mean(y)
	Vbad = mean((y-Ybad)**2)

	p = posterior(x, y, yvar, theta, bperp, Pbad, Ybad, Vbad)
	print 'starting p=', p

	chain = []
	oldp = p
	oldparams = (theta, bperp, Pbad, Ybad, Vbad)
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
			(theta, bperp, Pbad, Ybad, Vbad) = newparams
			newparams = (theta, bperp, 0, Ybad, Vbad)

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
	
	fig_width=5
	fig_height=5
	fig_size =	[fig_width,fig_height]
	params = {'axes.labelsize': 12,
			  'text.fontsize': 11,
			  'legend.fontsize': 12,
			  'xtick.labelsize':10,
			  'ytick.labelsize':10,
			  'text.usetex': True,
			  'figure.figsize': fig_size,
			  'image.interpolation':'nearest',
			  'image.origin':'lower',
			  }
	rcParams.update(params)

	# Plot data
	clf()
	ax = gca()
	for e in ellipses:
		ax.add_artist(e)
		e.set_facecolor('none')
	xlabel(r'$x$')
	ylabel(r'$y$')
	xlim(0,300)
	ylim(0,700)
	savefig(prefix + '-data.pdf')

	a = axis()
	xmin, xmax = xlim()
	ymin, ymax = ylim()
	xs = linspace(xmin, xmax, 2)
	Nchain = len(chain)
	if mixture:
		# select 10 samples at random from the second half of the chain.
		I = Nchain/2 + random.permutation(Nchain/2)[:10]
	else:
		I = array([argmax([p for (p, params) in chain])])
	for i in I:
		(p,params) = chain[i]
		(theta, bperp, Pbad, Ybad, Vbad) = params
		ys = tan(theta) * xs + bperp / cos(theta) # replace this with smarter linear algebra
		plot(xs, ys, color='k', alpha=0.3)
	axis(a)
	savefig(prefix + '-xy.pdf')

	if mixture:
		bgp = zeros(len(x))
		fgp = zeros(len(x))
		for (p,params) in chain[Nchain/2:]:
			(theta, bperp, Pbad, Ybad, Vbad) = params
			bgp += Pbad		 * single_point_likelihoods(x, y, yvar, theta, bperp, 1, Ybad, Vbad)
			fgp += (1.-Pbad) * single_point_likelihoods(x, y, yvar, theta, bperp, 0, Ybad, Vbad)
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
	# note horrifying theta = 0.5 * pi behavior!
	ms = array([tan(theta) for (p, (theta, bperp, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
	bs = array([bperp / cos(theta) for (p, (theta, bperp, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
	#plot(ms, bs, 'k,', alpha=0.1)
	xlabel('slope $m$')
	ylabel('intercept $b$')
	#savefig(prefix + '-mb-scatter.pdf')

	clf()
	(H, xe, ye) = histogram2d(ms, bs, bins=(100,100))
	imshow(log(1+H.T), extent=(xe.min(), xe.max(), ye.min(), ye.max()), aspect='auto',
		   cmap=antigray)
	xlabel('slope $m$')
	ylabel('intercept $b$')
	savefig(prefix + '-mb.pdf')

if __name__ == '__main__':
	args = sys.argv[1:]
	quick = 'quick' in args
	mixture = not('nomixture' in args)
	marginalize_mixture(mixture, quick)
