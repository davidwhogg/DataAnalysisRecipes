# haeringrix:
#   re-fit black-hole--bulge mass relationship
#
# license:
#   Copyright 2009 Dustin Lang and David W. Hogg.  All rights reserved.
#
# bugs:
#   - Packing and unpacking of params ugly and slow!
#

from numpy import *
import matplotlib
# need to use Agg to see minus signs (!) on some plots
matplotlib.use('Agg')
from matplotlib import rcParams
from matplotlib.patches import Ellipse
from pylab import *
# pylab must go before numpy.random import
import numpy.random as random
from numpy.linalg import *
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *
import cPickle as pickle
import sys
import string

def read_and_manipulate_data():
	x = table_fields('../data/Jahnke_haeringrixdata_new.fits')
	# approximation 1:
	# use the mid-points of the confidence intervals
	# NB x.errl_mbulge are positive...
	x.logmbulgemidpoint = 0.5 * (log10(x.mbulge + x.erru_mbulge) + log10(x.mbulge - x.errl_mbulge))
	x.logmbulgemidpoint_err = 0.5 * (log10(x.mbulge + x.erru_mbulge) - log10(x.mbulge - x.errl_mbulge))
	# ...while x.errl_mbh are negative!
	x.logmbhmidpoint = 0.5 * (log10(x.mbh + x.erru_mbh) + log10(x.mbh + x.errl_mbh))
	x.logmbhmidpoint_err = 0.5 * (log10(x.mbh + x.erru_mbh) - log10(x.mbh + x.errl_mbh))
	# make fitting information
	x.fitx = x.logmbulgemidpoint
	x.fity = x.logmbhmidpoint
	x.covar = zeros((len(x.mbh),2,2))
	for i in range(len(x.mbh)):
		x.covar[i,:,:] = diag([x[i].logmbulgemidpoint_err**2, x[i].logmbhmidpoint_err**2])
	return x

# returns an array of the single-point likelihoods.
# some good linear algebra would speed this up a lot
def single_point_likelihoods_varperp(x, y, covar, theta, bperp, varperp):
	projvar = zeros(len(x))
	projpos = zeros(len(x))
	unitv = array([-sin(theta),cos(theta)])
	for i,(xi,yi,covari) in enumerate(zip(x,y,covar)):
		projvar[i] = dot(unitv, dot(covari, unitv.T))
		projpos[i] = dot(unitv, array([xi,yi]).T)
	return ((1.0 / sqrt(2. * pi * (projvar + varperp)))
			* exp(-0.5 * (projpos - bperp)**2 / (projvar + varperp)))

def likelihood_varperp(data, params):
	(x, y, covar) = data
	theta = params[0]
	bperp = params[1]
	varperp = params[2]
	return prod(single_point_likelihoods_varperp(x, y, covar, theta, bperp, varperp))

# Technically an improper prior (just do naive range checks).  Feel free to insert your own.
def prior_varperp(params):
	varperp = params[2]
	return (varperp >= 0)

def posterior_varperp(data, params):
	return (likelihood_varperp(data, params)
			* prior_varperp(params))

# returns an array of the single-point likelihoods.
# some good linear algebra would speed this up a lot
def single_point_likelihoods_outlier(x, y, covar, theta, bperp, pbad, vbad):
	projvar = zeros(len(x))
	projpos = zeros(len(x))
	unitv = array([-sin(theta),cos(theta)])
	for i,(xi,yi,covari) in enumerate(zip(x,y,covar)):
		projvar[i] = dot(unitv, dot(covari, unitv.T))
		projpos[i] = dot(unitv, array([xi,yi]).T)
	return (((1.0 - pbad) / sqrt(2. * pi * projvar))
		* exp(-0.5 * (projpos - bperp)**2 / projvar)
		+ pbad / sqrt(2. * pi * (projvar + vbad))
		* exp(-0.5 * (projpos - bperp)**2 / (projvar + vbad)))

def likelihood_outlier(data, params):
	(x, y, covar) = data
	theta = params[0]
	bperp = params[1]
	pbad = params[2]
	vbad = params[3]
	return prod(single_point_likelihoods_outlier(x, y, covar, theta, bperp, pbad, vbad))

# Technically an improper prior (just do naive range checks).  Feel free to insert your own.
def prior_outlier(params):
	pbad = params[2]
	vbad = params[3]
	return (pbad > 0) * (pbad < 1) * (vbad > 0)

def posterior_outlier(data, params):
	return (likelihood_outlier(data, params)
		* prior_outlier(params))

def pick_new_parameters(step, params, dparams):
	indx = find(dparams > 0)
	indx = indx[step % len(indx)]
	newparams = params.copy()
	newparams[indx] += dparams[indx] * random.normal()
	return (newparams, indx)

def metropolis_chain(nsteps, data, params, dparams, posterior, pickfunk):
	p = posterior(data, params)
	print 'starting p=', p
	oldp = p
	oldparams = params
	bestparams = oldparams
	bestp = oldp
	print 'doing', nsteps, 'steps of MCMC...'
	chain = []
	step = 0
	npar = sum(dparams > 0)
	naccepts = zeros(params.shape)
	while step < nsteps:
		(newparams, indx) = pickfunk(step, oldparams, dparams)
		p = posterior(data, newparams)
		if p/oldp > random.uniform():
			# keep new parameters
			chain.append((p,newparams))
			oldparams = newparams
			oldp = p
			if p > bestp:
				bestp = p
				bestparams = newparams
			naccepts[indx] += 1
		else:
			# keep old parameters
			chain.append((oldp,oldparams))
		step += 1
		if (step % 10000 == 1):
			print step, (naccepts * npar / float(step)), bestp, bestparams
	print 'acceptance fraction', (naccepts * npar / float(step))
	return chain

# x,y both shape (N)
# invcovar shape (N,2,2)
def plot_covar(x, y, covar):
	ndata= len(x)
	ax = gca()
	ellipses=[]
	for ii in range(ndata):
		#Calculate the eigenvalues and the rotation angle
		eigs= linalg.eig(covar[ii,:,:])
		angle= arctan(-eigs[1][0,1]/eigs[1][1,1])/pi*180.
		thisellipse= Ellipse(array([x[ii],y[ii]]),2*sqrt(eigs[0][0]),
							 2*sqrt(eigs[0][1]),angle)
		ellipses.append(thisellipse)
		ax.add_artist(thisellipse)
		thisellipse.set_facecolor('none')

def plot_pickle(prefix):
	print "reading " + prefix + ".pickle"
	(x, chain) = pickle.loads(read_file(prefix + '.pickle'))
	probs		= array([p for (p,nil) in chain])
	thetas		= array([params[0] for (p,params) in chain])
	bperps		= array([params[1] for (p,params) in chain])
	thirdparams = array([params[2] for (p,params) in chain])
	slopes = tan(thetas)
	intercepts = bperps / cos(thetas)

	# set case-specifics
	plot_title = 'dummy plot title'
	if prefix == 'haeringrix-varperp':
		varperp_plots = True
		outlier_plots = False
		slopeone_plots = False
		plot_title = 'perpendicular variance model'
	if prefix == 'haeringrix-slopeone':
		varperp_plots = True
		outlier_plots = False
		slopeone_plots = True
		plot_title = 'perpendicular variance model, slope fixed at 1'
	if prefix == 'haeringrix-outlier':
		varperp_plots = False
		outlier_plots = True
		slopeone_plots = False
		plot_title = 'outlier rejection model'

	# measure intervals
	loindx = floor(0.025 * len(chain))
	hiindx = ceil(0.975 * len(chain))
	foo = argsort(slopes)
	slopelim = (slopes[foo[loindx]], slopes[foo[hiindx]])
	print "  %.2f < slope < %.2f" % slopelim
	if slopeone_plots:
		foo = argsort(intercepts)
		interceptlim = (intercepts[foo[loindx]], intercepts[foo[hiindx]])
		massratiolim = (10 ** (-interceptlim[1]), 10 ** (-interceptlim[0]))
		print "  %.0f < mass ratio < %.0f" % massratiolim
	foo = argsort(thirdparams)
	thirdlim = (thirdparams[foo[loindx]], thirdparams[foo[hiindx]])
	print "  %.2f < third parameter < %.2f" % thirdlim

	# plot the data
	plot(log10(x.mbulge), log10(x.mbh), 'k.')
	plot_covar(x.fitx, x.fity, x.covar)
	xlim((8.5, 12.5))
	xlabel('log bulge mass')
	ylim((6, 10))
	ylabel('log black hole mass')
	savefig(prefix+'-data.png')
	a = axis()

	# determine and label outliers
	if outlier_plots:
		nx = len(x.fitx)
		numerator = zeros(nx)
		denominator = zeros(nx)
		for (p, params) in chain[::101]:
			theta = params[0]
			bperp = params[1]
			pbad = params[2]
			vbad = params[3]
			numerator += (pbad
						  * single_point_likelihoods_outlier(x.fitx, x.fity, x.covar, theta, bperp, 1.0, vbad)
						  / single_point_likelihoods_outlier(x.fitx, x.fity, x.covar, theta, bperp, pbad, vbad))
			denominator += ones(nx)
		outlier_probs = numerator / denominator
		outlier_odds = outlier_probs / (1.0 - outlier_probs)
		for i, bgo in enumerate(outlier_odds):
			if bgo > 2.0:
				dyl = 0.01
				if x.mbulge[i] > (600 * x.mbh[i]):
					halign = 'left'
					dxl = 0.01
				else:
					halign = 'right'
					dxl = -0.01
				t = text(log10(x.mbulge[i])+dxl, log10(x.mbh[i])-dyl,
						 '%.1f' % bgo,
						 horizontalalignment=halign,
						 verticalalignment='top', color='r')
				t = text(log10(x.mbulge[i])+dxl, log10(x.mbh[i])+dyl,
						 string.join(string.split(x.name[i],'_'),' '),
						 horizontalalignment=halign,
						 verticalalignment='bottom', color='r')

	# plot a sample of lines
	(xmin, xmax) = xlim()
	(ymin, ymax) = ylim()
	xs = linspace(xmin, xmax, 2)
	# select 16 samples at random from the chain.
	I = random.permutation(len(chain))[:16]
	for i in I:
		ys = slopes[i] * xs + intercepts[i]
		plot(xs, ys, color='k', alpha=0.3)
	axis(a)
	title(plot_title)
	savefig(prefix + '-lines.png')

	# plot posterior sampling
	if not slopeone_plots:
		clf()
		hist(slopes[::11], 100, histtype='step', ec='k')
		slopeplotlim = (0.7,2.3)
		xlim(slopeplotlim)
		xlabel('slope')
		title(plot_title)
		axvline(slopelim[0], color='k', alpha=0.6)
		axvline(slopelim[1], color='k', alpha=0.6)
		savefig(prefix + '-slope.png')

	if varperp_plots:
		varperps = thirdparams
		clf()
		hist(varperps[::11], 100, histtype='step', ec='k')
		varperpplotlim = (0.0, 0.35)
		xlim(varperpplotlim)
		xlabel('perpendicular variance')
		title(plot_title)
		axvline(thirdlim[0], color='k', alpha=0.6)
		axvline(thirdlim[1], color='k', alpha=0.6)
		savefig(prefix + '-var.png')

		if not slopeone_plots:
			clf()
			plot(slopes[::101],
				 varperps[::101],
				 'k.', alpha=0.5)
			xlim(slopeplotlim)
			xlabel('slope')
			ylim(varperpplotlim)
			ylabel('perpendicular variance (dex)')
			title(plot_title)
			savefig(prefix + '-slopevar.png')

	if outlier_plots:
		clf()
		pbads = thirdparams
		vbads = array([params[3] for (p,params) in chain])
		plot(slopes[::101],
			 pbads[::101],
			 'k.', alpha=0.5)
		xlim(slopeplotlim)
		xlabel('slope')
		ylim((0.0,1.0))
		ylabel('outlier fraction')
		title(plot_title)
		savefig(prefix + '-slopepbad.png')

		clf()
		hist(pbads[::11], 100, histtype='step', ec='k')
		xlim((0.0,1.0))
		xlabel('outlier fraction')
		title(plot_title)
		axvline(thirdlim[0], color='k', alpha=0.6)
		axvline(thirdlim[1], color='k', alpha=0.6)
		savefig(prefix + '-pbad.png')

	if slopeone_plots:
		clf()
		hist(intercepts[::11], 100, histtype='step', ec='k')
		xlim((-3.2,-2.4))
		xlabel('intercept (log mass ratio in dex)')
		title(plot_title)
		axvline(interceptlim[0], color='k', alpha=0.6)
		axvline(interceptlim[1], color='k', alpha=0.6)
		savefig(prefix + '-int.png')

	# plot the chain
	clf()
	plot(log(probs[::101]),'k.', alpha=0.5)
	xlabel('link number / N')
	ylabel('ln posterior (nats)')
	title(plot_title)
	savefig(prefix + '-chain.png')

if __name__ == '__main__':
	prefix = 'haeringrix'
	args = sys.argv[1:]

	if 'outlier' in args:
		prefix += '-outlier'
		posterior = posterior_outlier
		params = array([0.9, -3.3, 0.2, 2.0]) # first guess
		dparams = array([0.01, 0.1, 0.3, 0.3]) # mcmc step sizes
		maxchain = 3e6
	elif 'slopeone' in args:
		prefix += '-slopeone'
		posterior = posterior_varperp
		params = array([arctan(1.0), -2.7/1.4, 0.1]) # first guess
		dparams = array([0, 0.1, 0.05]) # mcmc step sizes
		maxchain = 3e6
	else:
		prefix += '-varperp'
		posterior = posterior_varperp
		params = array([0.9, -3.3, 0.1]) # first guess
		dparams = array([0.01, 0.1, 0.05]) # mcmc step sizes
		maxchain = 3e6

	# plot only?
	if 'plotonly' in args:
		plot_pickle(prefix)
		sys.exit(0)

	# read data (and write minimal pickle)
	x = read_and_manipulate_data()

	# fit data by MCMC
	burnchain = metropolis_chain(20002, (x.fitx, x.fity, x.covar),
								 params, dparams, posterior, pick_new_parameters)
	interval = 11
	chain = burnchain[-1:]
	while (interval * (len(chain) + 1)) < maxchain:
		i = int(random.uniform(high=len(chain)))
		(nil, params) = chain[i]
		thischain = metropolis_chain(int(maxchain/9.99), (x.fitx, x.fity, x.covar),
								  params, dparams, posterior, pick_new_parameters)
		chain += thischain[::interval]
		print 'writing ' + prefix + '.pickle'
		write_file(pickle.dumps((x, chain), pickle.HIGHEST_PROTOCOL), prefix + '.pickle')
		plot_pickle(prefix)
