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
import math as m
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.stats as stats
from generate_data import read_data
import matplotlib
matplotlib.use('Agg')
from pylab import *
from matplotlib.pyplot import *
from matplotlib import rc
import bovy_plot as plot
from matplotlib.patches import Ellipse


def ex13(exclude=sc.array([1,2,3,4]),plotfilename='ex13.png',
         nburn=1000,nsamples=100000,
         parsigma=[1,m.pi/200.,.01,.5,1.,.05,.1,.005],
		 bovyprintargs={}):
    """ex13: solve exercise 13 by MCMC
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
       nburn          - number of burn-in samples
       nsamples       - number of samples to take after burn-in
       parsigma       - proposal distribution width (Gaussian)
    Output:
       plot
    History:
       2010-05-06 - Written - Bovy (NYU)
    """
    #Read the data
    data= read_data('data_allerr.dat',allerr=True)
    ndata= len(data)
    nsample= ndata- len(exclude)
    #First find the chi-squared solution, which we will use as an
    #initial gues
    #Put the dat in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    A= sc.ones((nsample,2))
    C= sc.zeros((nsample,nsample))
    Z= sc.zeros((nsample,2))
    yerr= sc.zeros(nsample)
    ycovar= sc.zeros((2,nsample,2))#Makes the sc.dot easier
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
            pass
        else:
            Y[jj]= data[ii][1][1]
            X[jj]= data[ii][1][0]
            Z[jj,0]= X[jj]
            Z[jj,1]= Y[jj]
            A[jj,1]= data[ii][1][0]
            C[jj,jj]= data[ii][2]**2.
            yerr[jj]= data[ii][2]
            ycovar[0,jj,0]= data[ii][3]**2.
            ycovar[1,jj,1]= data[ii][2]**2.
            ycovar[0,jj,1]= data[ii][4]*m.sqrt(ycovar[0,jj,0]*ycovar[1,jj,1])
            ycovar[1,jj,0]= ycovar[0,jj,1]
            jj= jj+1
    #Now compute the best fit and the uncertainties
    bestfit= sc.dot(linalg.inv(C),Y.T)
    bestfit= sc.dot(A.T,bestfit)
    bestfitvar= sc.dot(linalg.inv(C),A)
    bestfitvar= sc.dot(A.T,bestfitvar)
    bestfitvar= linalg.inv(bestfitvar)
    bestfit= sc.dot(bestfitvar,bestfit)
    #Now sample
    inittheta= m.acos(1./m.sqrt(1.+bestfit[1]**2.))
    if bestfit[1] < 0.:
        inittheta= m.pi- inittheta
    initialguess= sc.array([m.cos(inittheta),inittheta,0.,sc.mean(X),sc.mean(Y),m.log(sc.var(X)),m.log(sc.var(X)),0.])#(m,b,Pb,Yb,Vb)
    #With this initial guess start off the sampling procedure
    initialX= objective(initialguess,Z,ycovar)
    currentX= initialX
    bestX= initialX
    bestfit= initialguess
    currentguess= initialguess
    naccept= 0
    samples= []
    samples.append(currentguess)
    for jj in range(nburn+nsamples):
        #Draw a sample from the proposal distribution
        newsample= sc.zeros(8)
        newsample[0]= currentguess[0]+stats.norm.rvs()*parsigma[0]
        newsample[1]= currentguess[1]+stats.norm.rvs()*parsigma[1]
        newsample[2]= currentguess[2]+stats.norm.rvs()*parsigma[2]
        newsample[3]= currentguess[3]+stats.norm.rvs()*parsigma[3]
        newsample[4]= currentguess[4]+stats.norm.rvs()*parsigma[4]
        newsample[5]= currentguess[5]+stats.norm.rvs()*parsigma[5]
        newsample[6]= currentguess[6]+stats.norm.rvs()*parsigma[6]
        newsample[7]= currentguess[7]+stats.norm.rvs()*parsigma[7]
        #Calculate the objective function for the newsample
        newX= objective(newsample,Z,ycovar)
        #Accept or reject
        #Reject with the appropriate probability
        u= stats.uniform.rvs()
        try:
            test= m.exp(newX-currentX)
        except OverflowError:
            test= 2.
        if u < test:
            #Accept
            currentX= newX
            currentguess= newsample
            naccept= naccept+1
        if currentX > bestX:
            bestfit= currentguess
            bestX= currentX
        samples.append(currentguess)
    if double(naccept)/(nburn+nsamples) < .5 or double(naccept)/(nburn+nsamples) > .8:
        print "Acceptance ratio was "+str(double(naccept)/(nburn+nsamples))

    samples= sc.array(samples).T[:,nburn:-1]
    print "Best-fit, overall"
    print bestfit, sc.mean(samples[2,:]), sc.median(samples[2,:])

    histmb,edges= sc.histogramdd(samples.T[:,0:2],bins=round(sc.sqrt(nsamples)/2.))
    indxi= sc.argmax(sc.amax(histmb,axis=1))
    indxj= sc.argmax(sc.amax(histmb,axis=0))
    print "Best-fit, marginalized"
    print edges[0][indxi-1], edges[1][indxj-1]
    print edges[0][indxi], edges[1][indxj]
    print edges[0][indxi+1], edges[1][indxj+1]

    t= edges[1][indxj]
    bcost= edges[0][indxi]
    mf= m.sqrt(1./m.cos(t)**2.-1.)
    b= bcost/m.cos(t)
    print b, mf

    #Plot result
    plot.bovy_print(**bovyprintargs)
    xrange=[0,300]
    yrange=[0,700]
    plot.bovy_plot(sc.array(xrange),mf*sc.array(xrange)+b,
                   'k-',xrange=xrange,yrange=yrange,
                   xlabel=r'$x$',ylabel=r'$y$',zorder=2)
    for ii in range(10):
        #Random sample
        ransample= sc.floor((stats.uniform.rvs()*nsamples)).astype('int')
        ransample= samples.T[ransample,0:2]
        mf= m.sqrt(1./m.cos(ransample[1])**2.-1.)
        b= ransample[0]/m.cos(ransample[1])
        bestb= b
        bestm= mf
        plot.bovy_plot(sc.array(xrange),bestm*sc.array(xrange)+bestb,
                       overplot=True,color='0.75',zorder=0)

    #Add labels
    nsamples= samples.shape[1]
    for ii in range(nsample):
        Pb= 0.
        for jj in range(nsamples):
            Pb+= Pbad(samples[:,jj],Z[ii,:],ycovar[:,ii,:])
        Pb/= nsamples
        text(Z[ii,0]+5,Z[ii,1]+5,'%.1f'%Pb,color='0.5',zorder=3)


    #Plot the data OMG straight from plot_data.py
    data= read_data('data_allerr.dat',True)
    ndata= len(data)
    #Create the ellipses and the data points
    id= sc.zeros(nsample)
    x= sc.zeros(nsample)
    y= sc.zeros(nsample)
    ellipses=[]
    ymin, ymax= 0, 0
    xmin, xmax= 0,0
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
            continue
        id[jj]= data[ii][0]
        x[jj]= data[ii][1][0]
        y[jj]= data[ii][1][1]
        #Calculate the eigenvalues and the rotation angle
        ycovar= sc.zeros((2,2))
        ycovar[0,0]= data[ii][3]**2.
        ycovar[1,1]= data[ii][2]**2.
        ycovar[0,1]= data[ii][4]*m.sqrt(ycovar[0,0]*ycovar[1,1])
        ycovar[1,0]= ycovar[0,1]
        eigs= linalg.eig(ycovar)
        angle= m.atan(-eigs[1][0,1]/eigs[1][1,1])/m.pi*180.
        thisellipse= Ellipse(sc.array([x[jj],y[jj]]),2*m.sqrt(eigs[0][0]),
                             2*m.sqrt(eigs[0][1]),angle)
        ellipses.append(thisellipse)
        if (x[jj]+m.sqrt(ycovar[0,0])) > xmax:
            xmax= (x[jj]+m.sqrt(ycovar[0,0]))
        if (x[jj]-m.sqrt(ycovar[0,0])) < xmin:
            xmin= (x[jj]-m.sqrt(ycovar[0,0]))
        if (y[jj]+m.sqrt(ycovar[1,1])) > ymax:
            ymax= (y[jj]+m.sqrt(ycovar[1,1]))
        if (y[jj]-m.sqrt(ycovar[1,1])) < ymin:
            ymin= (y[jj]-m.sqrt(ycovar[1,1]))
        jj= jj+1
        
    #Add the error ellipses
    ax=gca()
    for e in ellipses:
        ax.add_artist(e)
        e.set_facecolor('none')
    ax.plot(x,y,color='k',marker='o',linestyle='None')


    plot.bovy_end_print(plotfilename)



def objective(pars,Z,ycovar):
    """The objective function"""
    bcost= pars[0]
    t= pars[1]
    Pb= pars[2]
    Xb= pars[3]
    Yb= pars[4]
    Zb= sc.array([Xb,Yb])
    Vb1= sc.exp(pars[5])
    Vb2= sc.exp(pars[6])
    corr= pars[7]
    V= sc.array([[Vb1,sc.sqrt(Vb1*Vb2)*corr],[sc.sqrt(Vb1*Vb2)*corr,Vb2]])
    v= sc.array([-sc.sin(t),sc.cos(t)])
    if Pb < 0. or Pb > 1.:
        return -sc.finfo(sc.dtype(sc.float64)).max
    if corr < -1. or corr > 1.:
        return -sc.finfo(sc.dtype(sc.float64)).max
    delta= sc.dot(v,Z.T)-bcost
    sigma2= sc.dot(v,sc.dot(ycovar,v))

    ndata= Z.shape[0]
    detVycovar= sc.zeros(ndata)
    deltaOUT= sc.zeros(ndata)
    for ii in range(ndata):
        detVycovar[ii]= m.sqrt(linalg.det(V+ycovar[:,ii,:]))
        deltaOUT[ii]= sc.dot(Z[ii,:]-Zb,sc.dot(linalg.inv(V+ycovar[:,ii,:]),Z[ii,:]-Zb))
    return sc.sum(sc.log((1.-Pb)/sc.sqrt(2.*m.pi*sigma2/sc.cos(t)**2.)*
                         sc.exp(-0.5*delta**2./sigma2)
                         +Pb/2./m.pi/detVycovar
                         *sc.exp(-0.5*deltaOUT)))

def Pbad(pars,Z,ycovar):
    """Probability that one data point is bad"""
    bcost= pars[0]
    t= pars[1]
    Pb= pars[2]
    Xb= pars[3]
    Yb= pars[4]
    Zb= sc.array([Xb,Yb])
    Vb1= sc.exp(pars[5])
    Vb2= sc.exp(pars[6])
    corr= pars[7]
    V= sc.array([[Vb1,sc.sqrt(Vb1*Vb2)*corr],[sc.sqrt(Vb1*Vb2)*corr,Vb2]])
    v= sc.array([-sc.sin(t),sc.cos(t)])
    delta= sc.dot(v,Z)-bcost
    sigma2= sc.dot(v,sc.dot(ycovar,v))

    detVycovar= m.sqrt(linalg.det(V+ycovar[:,:]))
    deltaOUT= sc.dot(Z-Zb,sc.dot(linalg.inv(V+ycovar[:,:]),Z-Zb))
    Pbad= Pb/2./m.pi/detVycovar*sc.exp(-0.5*deltaOUT)
    return Pbad/(Pbad+(1.-Pb)/sc.sqrt(2.*m.pi*sigma2)*sc.exp(-0.5*delta**2./sigma2/sc.cos(t)**2.))
