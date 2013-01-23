#############################################################################
#Copyright (c) 2010, Jo Bovy, David W. Hogg, Dustin Lang
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without 
#modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, 
#      this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice, 
#      this list of conditions and the following disclaimer in the 
#      documentation and/or other materials provided with the distribution.
#   The name of the author may not be used to endorse or promote products 
#      derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
#WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.
#############################################################################
import scipy as sc
import math as ma
import math
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.stats as stats
import copy as c
from generate_data import read_data
import matplotlib
matplotlib.use('Agg')
from pylab import *
from matplotlib.pyplot import *
from matplotlib import rc

MAXP= 0.999

def lnposterior(x,y,yivar,m,b,qlist,pgood,bgmean,bgivar):
    return lnprior(m,b,qlist,pgood)+lnlikelihood(x,y,yivar,m,b,qlist,pgood,bgmean,bgivar)

def lnlikelihood(x,y,yivar,m,b,qlist,pgood,bgmean,bgivar):
    lnl = sc.sum(qlist*0.5*(sc.log(yivar)-ma.log(2.*math.pi)))-sc.sum(qlist*0.5*yivar*(y-m*x-b)**2)
    lnl += sc.sum((1-qlist)*0.5*(sc.log(bgivar)-ma.log(2.*math.pi)))-sc.sum((1-qlist)*0.5*bgivar*(y-bgmean)**2)
    return lnl

def lnprior(m,b,qlist,pgood):
    lnp = sc.sum(qlist*sc.log(pgood)+(1-qlist)*sc.log(1.0-pgood))
    if sc.sum(qlist) < 2: lnp = 100.0*sc.log(1.0-MAXP)
    return lnp

def ex8(plotfilename='ex8.png',nburn=1000,nsamples=10000,parsigma=[.075,2.,0.1]):
    """ex8: solve exercise 8 using...?
    Input:
       plotfilename   - filename for the output plot
       nburn          - number of burn-in samples
       nsamples       - number of samples to take after burn-in
       parsigma       - proposal distribution width (Gaussian)
    Output:
       plot
    History:
       2009-06-25 -- hacked from Bovy code - Hogg (NYU)
    """
    sc.random.seed(-1) #In the interest of reproducibility (if that's a word)
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    #Put the data in the appropriate arrays and matrices
    X= sc.zeros(ndata)
    Y= sc.zeros(ndata)
    A= sc.ones((ndata,2))
    Yivar= sc.zeros(ndata)
    C= sc.zeros((ndata,ndata))
    yerr= sc.zeros(ndata)
    jj= 0
    for ii in range(ndata):
        X[jj]= data[ii][1][0]
        Y[jj]= data[ii][1][1]
        A[jj,1]= data[ii][1][0]
        Yivar[jj]= 1.0/(data[ii][2]**2)
        C[jj,jj]= data[ii][2]**2
        yerr[jj]= data[ii][2]
        jj= jj+1
    #First find the chi-squared solution, which we will use as an
    #initial guess
    bestfit= sc.dot(linalg.inv(C),Y.T)
    bestfit= sc.dot(A.T,bestfit)
    bestfitvar= sc.dot(linalg.inv(C),A)
    bestfitvar= sc.dot(A.T,bestfitvar)
    bestfitvar= linalg.inv(bestfitvar)
    bestfit= sc.dot(bestfitvar,bestfit)
    m= bestfit[1]
    b= bestfit[0]
    q= sc.array([1 for cc in range(ndata)])
    q[0:4] = 0
    pgood= 0.9
    #pgood=0.999759#3 sigma for uncertainty~50
    initialguess= [m,b,q,pgood]
    print initialguess
    #With this initial guess start off the sampling procedure
    bgmean= sc.mean(Y)
    bgivar= 1.0/sc.sum((Y-bgmean)**2)
    initialX= lnposterior(X,Y,Yivar,m,b,q,pgood,bgmean,bgivar)
    currentX= initialX
    bestX= initialX
    bestfit= initialguess
    currentguess= initialguess
    naccept= 0
    for jj in range(nburn+nsamples):
        #Draw a sample from the proposal distribution
        thisguess = c.deepcopy(currentguess)
        m= thisguess[0]
        b= thisguess[1]
        q= thisguess[2]
        pgood= thisguess[3]
        #First Gibbs sample each q
        for ii in range(ndata):
            thisdatagood= ma.sqrt(Yivar[ii]/(2.*math.pi))*ma.exp(-.5*(Y[ii]-m*X[ii]-b)**2.*Yivar[ii])*pgood
            thisdatabad= ma.sqrt(bgivar/(2.*math.pi))*ma.exp(-.5*(Y[ii]-bgmean)**2.*bgivar)*(1.0-pgood)
            a= thisdatagood/(thisdatagood+thisdatabad)
            u= stats.uniform.rvs()
            if u<a:
                q[ii]= 1
            else:
                q[ii]= 0
        #Then Metropolis sample m and b
        m += stats.norm.rvs()*parsigma[0]
        b += stats.norm.rvs()*parsigma[1]
        pgood += stats.norm.rvs()*parsigma[2]
        if pgood > MAXP: pgood = MAXP
        if pgood < (1.0-MAXP): pgood = (1.0-MAXP)
        newsample = [m,b,q,pgood]
        #Calculate the objective function for the newsample
        newX= lnposterior(X,Y,Yivar,m,b,q,pgood,bgmean,bgivar)
        #Accept or reject
        #Reject with the appropriate probability
        u= stats.uniform.rvs()
        if u < ma.exp(newX-currentX):
            #Accept
            currentX= newX
            currentguess= newsample
            naccept= naccept+1
        if currentX > bestX:
            print currentguess
            bestfit= currentguess
            bestX= currentX
    print "Acceptance ratio was "+str(double(naccept)/(nburn+nsamples))

    #Now plot the best solution
    fig_width=5
    fig_height=5
    fig_size =  [fig_width,fig_height]
    params = {'axes.labelsize': 12,
              'text.fontsize': 11,
              'legend.fontsize': 12,
              'xtick.labelsize':10,
              'ytick.labelsize':10,
              'text.usetex': True,
              'figure.figsize': fig_size}
    rcParams.update(params)
    #Plot data
    errorbar(X,Y,yerr,color='k',marker='o',color='k',linestyle='None')
    xlabel(r'$x$')
    ylabel(r'$y$')
    xlim(0,300)
    ylim(0,700)
    xmin, xmax= xlim()
    (m,b,q,pgood) = bestfit
    print bestfit
    print m
    xs= sc.linspace(xmin,xmax,3)
    ys= m*xs+b
    if b < 0:
        sgn_str= '-'
    else:
        sgn_str= '+'
    label= r'$y = %4.2f\, x'% m+sgn_str+ '%4.0f ' % ma.fabs(b)+'$'#+r'; X = '+ '%3.1f' % bestX+'$'
    plot(xs,ys,color='k',ls='--',label=label)
    l=legend(loc=(.3,.1),numpoints=8)
    l.draw_frame(False)
    xlim(0,300)
    ylim(0,700)
    savefig(plotfilename,format='png')
    
    return 0

