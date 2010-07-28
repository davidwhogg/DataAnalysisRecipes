# code by Hogg (NYU) and Lang (Princeton)
# Copyright 2009, 2010 Hogg & Lang
# for now, all rights reserved

# to-do items
# -----------
# - switch non-parametric model over to linear interpolation
#   - this requires synchronized changes to the chi2 *and* wls functions
# - write gs command that builds multi-page pdf files, of the form
#   - gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=foo.pdf poly-fits*.pdf

from numpy import *
import matplotlib
matplotlib.use('Agg')
from pylab import *
import numpy.random as random
import os

plot_format = '.pdf'
gscmd = 'pdfjoin --outfile '
title_prefix = ''
random.seed(23) # 42
trueorder = 4
true_a_val = random.normal(size=trueorder+1)
data_N = 20
data_x = random.uniform(low=-1, high=1, size=data_N)

def polynomial(x, a):
	y = zeros_like(x)
	for i,ai in enumerate(a):
		y += ai*x**i
	return y

data_y = polynomial(data_x, true_a_val)
data_sigmay = random.uniform(low=0.1, high=0.4, size=data_y.shape)
data_y += random.normal(size=data_y.shape) * data_sigmay
xlimits = [min(data_x-0.1),max(data_x+0.1)]
ylimits = [min(data_y-data_sigmay-0.2),max(data_y+data_sigmay+0.2)]

def ln_gaussian_1d(x, mx, varx):
	detvar = varx
	invvar = 1./varx
	dx = x-mx
	d2 = (dx * invvar * dx)
	return -0.5*log(2.*pi*detvar) -0.5 * d2

def true_a():
	return true_a_val

def get_data_no_outliers():
	return (data_x, data_y, data_sigmay)

# Plot data with error bars, standard axis limits, etc.
def plot_yerr(x, y, sigmay):
	# plot data with error bars
	errorbar(x, y, yerr=sigmay, fmt='.', ms=7, lw=1, color='k')
	# if you put '$' in you can make Latex labels
	xlabel('$x$')
	ylabel('$y$')
	xlim(*xlimits)
	ylim(*ylimits)
	title(title_prefix)

def plot_poly(a, **kwargs):
	x = linspace(xlimits[0], xlimits[1], 1000)
	y = polynomial(x, a)
	if not 'alpha' in kwargs:
		kwargs['alpha'] = 0.5
	if not 'lw' in kwargs:
		kwargs['lw'] = 2.
	p = plot(x, y, 'k-', **kwargs)
	return p

def poly_chi2(x, y, sigmay, a):
	return sum(((y - polynomial(x, a)) / sigmay)**2)

# Weighted least squares method.  Returns best-fit "a"
def poly_wls(x, y, sigmay, order):
	N = len(x)
	Cinv = diag(1./sigmay**2)
	X = zeros((N,order+1))
	for i in range(order+1):
		X[:,i] = x**i
	XTCinvX = dot(dot(X.T, Cinv), X)
	XTCinvy = dot(dot(X.T, Cinv), y)
	beta = dot(inv(XTCinvX), XTCinvy)
	print 'beta', beta
	return beta

def ln_loo_stiffline(x, y, sigmay, xa, epsilon, iout):
	I = (arange(len(x)) != iout)
	a = stiffline_wls(x[I], y[I], sigmay[I], xa, epsilon)
	yiout = stiffline(array([x[iout]]), xa, a)
	return (a, ln_gaussian_1d(y[iout], yiout, sigmay[iout]**2))

def ln_loo_poly(x, y, sigmay, order, iout):
	I = (arange(len(x)) != iout)
	a = poly_wls(x[I], y[I], sigmay[I], order)
	yiout = polynomial(x[iout], a)
	return (a, ln_gaussian_1d(y[iout], yiout, sigmay[iout]**2))

def plot_stiffline(xa, a, **kwargs):
	if not 'alpha' in kwargs:
		kwargs['alpha'] = 0.5
	if not 'lw' in kwargs:
		kwargs['lw'] = 2.
	p = plot(xa, a, 'k-', **kwargs)
	return p

def stiffline(x, xa, a):
	return a[stiffline_match(x, xa)]

def stiffline_match(x, xa):
	matcha= zeros_like(x.astype(int))
	for ii,xx in enumerate(x):
		matcha[ii] = argmin((xa-xx)**2)
	return matcha

def stiffline_chi2(x, y, sigmay, xa, a, epsilon):
	d = 1
	ymodel = stiffline(x, xa, a)
	chi2 = sum(((y - ymodel) / sigmay)**2)
	chi2 += epsilon * sum((a[:-d] - a[d:])**2)
	return chi2

# stupid-ass iterative WLS algo; returns best-fit "a"
def stiffline_wls(x, y, sigmay, xa, epsilon):
	d = 1
	maxiter = 1000
	tolerance = 1e-5
	matcha = stiffline_match(x, xa)
	# initialize
	na = len(xa)
	a = zeros(na) + mean(y)
	iter = 0
	dchi2 = 100 * tolerance
	chi2 = stiffline_chi2(x, y, sigmay, xa, a, epsilon)
	while (iter < maxiter) and (dchi2 > tolerance):
		oldchi2 = chi2
		for j in range(na):
			numerator = 0.0
			denominator = 0.0
			if (j-d) >= 0:
				numerator += epsilon * a[j-d]
				denominator += epsilon
			if (j+d) < na:
				numerator += epsilon * a[j+d]
				denominator += epsilon
			for (ii,jj) in enumerate(matcha):
				if jj == j:
					numerator += y[ii] / sigmay[ii]**2
					denominator += 1.0 / sigmay[ii]**2
			a[j] = numerator / denominator
		chi2 = stiffline_chi2(x, y, sigmay, xa, a, epsilon)
		dchi2 = oldchi2 - chi2
		iter += 1
	print 'iteration', iter
	print '  chi2', chi2
	print '  dchi2', dchi2
	return a

def main_black():
	print 'making black page, you tool'
	axes([0,0,1,1], axisbg='k')
	savefig('black' + plot_format)

def main_stiffline():
	prefix = 'stiffline-'
	(x, y, sigmay) = get_data_no_outliers()
	xa = linspace(xlimits[0], xlimits[1], 40)
	log2epslist = (range(12))[::-1]
	nepsilon = len(log2epslist)
	ln_crossval_like = zeros(nepsilon)
	for ii,log2eps in enumerate(log2epslist):
		epsilon = 2.0**log2eps
		print 'working on epsilon = %f' % epsilon

		clf()
		plot_yerr(x, y, sigmay)
		plot_stiffline(xa, stiffline_wls(x, y, sigmay, xa, epsilon))
		xlim(*xlimits)
		ylim(*ylimits)
		title('$K = %i$ ; $\log_2(\epsilon) =  %i$' % (len(xa), log2eps));
		savefig(prefix + 'log2eps-%02i' % ii + plot_format)

		loo = [ln_loo_stiffline(x, y, sigmay, xa, epsilon, iout) for iout in range(len(x))]
		ln_crossval_like[ii] = sum([lnlike for (a,lnlike) in loo])

		clf()
		plot_yerr(x, y, sigmay)
		for a,nil in loo:
			plot_stiffline(xa, a)
		xlim(*xlimits)
		ylim(*ylimits)
		title('$K = %i$ ; $\log_2(\epsilon) =  %i$' % (len(xa), log2eps));
		savefig(prefix + 'fits-%02i' % ii + plot_format)

	clf()
	mx = max(ln_crossval_like)
	I = argsort(-ln_crossval_like)
	plot(maximum(mx-1000, ln_crossval_like), 'ko-')
	ylim(mx-20, mx+2)
	xlim(-1, 15)
	xlabel('$\log_2(\epsilon)$')
	ylabel('cross-validation log-likelihood')
	title('$K = %i$' % len(xa))
	savefig(prefix + 'crossval' + plot_format)

	cmdstr = gscmd + '%slog2eps.pdf %slog2eps-*.pdf' % (prefix, prefix)
	print os.system(cmdstr)
	cmdstr = gscmd + '%sfits.pdf %sfits-*.pdf' % (prefix, prefix)
	print os.system(cmdstr)
	cmdstr = gscmd + '%s.pdf black.pdf poly-data.pdf poly-truth.pdf %slog2eps.pdf %sfits.pdf %scrossval.pdf' % (prefix, prefix, prefix, prefix)
	print os.system(cmdstr)
	return

def main_poly():
	(x, y, sigmay) = get_data_no_outliers()

	for efrac,prefix,cvtit in [(1,'poly-',''), (0.6,'poly-wrong-', 'errors underestimated by 40 percent')]:
		clf()
		plot_yerr(x, y, sigmay*efrac)
		xlim(*xlimits)
		ylim(*ylimits)
		savefig(prefix + 'data' + plot_format)
		clf()
		plot_yerr(x, y, sigmay*efrac)
		plot_poly(true_a())
		xlim(*xlimits)
		ylim(*ylimits)
		title('truth: order %i' % (len(true_a())-1))
		savefig(prefix + 'truth' + plot_format)

		maxorder = 16
		chi2 = zeros(maxorder)
		ln_crossval_like = zeros(maxorder)
		for order in range(maxorder):
			clf()
			plot_yerr(x, y, sigmay*efrac)
			a = poly_wls(x, y, sigmay*efrac, order)
			plot_poly(a)
			chi2[order] = poly_chi2(x, y, sigmay*efrac, a)
			xlim(*xlimits)
			ylim(*ylimits)
			title('order %i ; $K=%i$' % (order, order+1))
			savefig(prefix + 'order-%02i' % order + plot_format)

			loo = [ln_loo_poly(x, y, sigmay*efrac, order, iout) for iout in range(len(x))]
			ln_crossval_like[order] = sum([lnlike for (a,lnlike) in loo])

			clf()
			plot_yerr(x, y, sigmay*efrac)
			for a,nil in loo:
				plot_poly(a)
			xlim(*xlimits)
			ylim(*ylimits)
			title('order %i ; $K=%i$' % (order, order+1))
			savefig(prefix + 'fits-%02i' % order + plot_format)

		clf()
		mx = max(ln_crossval_like)
		I = argsort(-ln_crossval_like)
		print ln_crossval_like[I]
		plot(maximum(mx-1000, ln_crossval_like), 'ko-')
		ylim(mx-20, mx+2)
		xlim(-1, 15)
		axvline(trueorder, color='k', alpha=0.5, lw=2.)
		xlabel('polynomial order')
		ylabel('cross-validation log-likelihood')
		title(cvtit)
		savefig(prefix + 'crossval' + plot_format)

		clf()
		mx = chi2[trueorder]
		plot(chi2, 'ko-')
		ylim(mx-10, mx+10)
		xlim(-1, 15)
		axvline(trueorder, color='k', alpha=0.5, lw=2.)
		xlabel('polynomial order')
		ylabel('$\\chi^2$')
		title(cvtit)
		savefig(prefix + 'chi2' + plot_format)

		clf()
		aic = chi2 + 2.0*(arange(maxorder)+1.0)
		mx = min(aic)
		plot(aic, 'ko-')
		ylim(mx-2, mx+20)
		xlim(-1, 15)
		axvline(trueorder, color='k', alpha=0.5, lw=2.)
		xlabel('polynomial order')
		ylabel('AIC $\\chi^2+2\\,K$')
		title(cvtit)
		savefig(prefix + 'aic' + plot_format)

		clf()
		bic = chi2 + log(len(x))*(arange(maxorder)+1.0)
		mx = min(bic)
		plot(bic, 'ko-')
		ylim(mx-2, mx+20)
		xlim(-1, 15)
		axvline(trueorder, color='k', alpha=0.5, lw=2.)
		xlabel('polynomial order')
		ylabel('BIC $\\chi^2+K\\,\\ln(N)$')
		title(cvtit)
		savefig(prefix + 'bic' + plot_format)

		cmdstr = gscmd + '%sorder.pdf %sorder-*.pdf' % (prefix, prefix)
		print os.system(cmdstr)
		cmdstr = gscmd + '%sfits.pdf %sfits-*.pdf' % (prefix, prefix)
		print os.system(cmdstr)
		cmdstr = gscmd + '%s.pdf black.pdf %sdata.pdf %struth.pdf %sorder.pdf %schi2.pdf %saic.pdf %sbic.pdf %scrossval.pdf' % (prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix)
		print os.system(cmdstr)
	return

if __name__ == '__main__':
	args = sys.argv[1:]
	if 'black' in args:
		main_black()
		sys.exit(0)
	main_black()
	main_poly()
	main_stiffline()
	cmdstr = gscmd + 'crossvalidate.pdf poly-.pdf poly-fits-*.pdf poly-crossval.pdf poly-wrong-.pdf stiffline-.pdf black.pdf'
	print os.system(cmdstr)
