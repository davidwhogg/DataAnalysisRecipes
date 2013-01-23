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

def ex7a(exclude=sc.array([1,2,3,4]),plotfilename='ex7a.png'):
    """ex7a: solve exercise 7 using non-linear optimization
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
    Output:
       plot
    History:
       2009-06-01 - Written - Bovy (NYU)
    """
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    nsample= ndata- len(exclude)
    #First find the chi-squared solution, which we will use as an
    #initial gues for the bi-exponential optimization
    #Put the dat in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    A= sc.ones((nsample,2))
    C= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
            pass
        else:
            Y[jj]= data[ii][1][1]
            X[jj]= data[ii][1][0]
            A[jj,1]= data[ii][1][0]
            C[jj,jj]= data[ii][2]**2.
            yerr[jj]= data[ii][2]
            jj= jj+1
    #Now compute the best fit and the uncertainties
    bestfit= sc.dot(linalg.inv(C),Y.T)
    bestfit= sc.dot(A.T,bestfit)
    bestfitvar= sc.dot(linalg.inv(C),A)
    bestfitvar= sc.dot(A.T,bestfitvar)
    bestfitvar= linalg.inv(bestfitvar)
    bestfit= sc.dot(bestfitvar,bestfit)
    initialguess= sc.array([bestfit[0],bestfit[1]])
    #Now optimize the soft chi-squared objective function
    Qs= [1.,2.]
    bestfitssoft= sc.zeros((2,len(Qs)))
    chisqQ= sc.zeros(len(Qs))
    for ii in range(len(Qs)):
        print "Working on Q = "+str(Qs[ii])
        bestfitsoft1= optimize.fmin(softchisquared,initialguess,(X,Y,yerr,Qs[ii]),disp=False)
        #Restart the optimization once using a different method
        bestfitsoft= optimize.fmin_powell(softchisquared,bestfitsoft1,(X,Y,yerr,Qs[ii]),disp=False)
        if linalg.norm(bestfitsoft-bestfitsoft1) > 10**-12:
            if linalg.norm(bestfitsoft-bestfitsoft1) < 10**-6:
                print "Different optimizers give slightly different results..."
            else:
                print "Different optimizers give rather different results..."
            print "The norm of the results differs by %g" % linalg.norm(bestfitsoft-bestfitsoft1)
            try:
                x=raw_input('continue to plot? [yn]\n')
            except EOFError:
                print "Since you are in non-interactive mode I will assume 'y'"
                x='y'
            if x == 'n':
                    print "returning..."
                    return -1
        bestfitssoft[:,ii]= bestfitsoft
        #Calculate chi^2_Q
        for jj in range(nsample):
            chisqQ[ii]= chisqQ[ii]+1./(yerr[jj]**2/(Y[jj]-X[jj]*bestfitsoft[1]-bestfitsoft[1])**2+1./Qs[ii]**2)
    
    #Now plot the solution
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
    #Plot the best fit line for the different Qs
    linestyles= ('--',':', '-.')
    for jj in range(len(Qs)):
        xlim(0,300)
        ylim(0,700)
        xmin, xmax= xlim()
        nsamples= 1001
        xs= sc.linspace(xmin,xmax,nsamples)
        ys= sc.zeros(nsamples)
        for ii in range(nsamples):
            ys[ii]= bestfitssoft[0,jj]+bestfitssoft[1,jj]*xs[ii]
            if bestfitssoft[0,jj] < 0:
                sgn_str= '-'
            else:
                sgn_str= '+'
        label= r'$Q= '+'%i: y = %4.2f\, x'% (Qs[jj], bestfitssoft[1,jj]) +sgn_str+ '%4.0f ' % m.fabs(bestfitssoft[0,jj])+r'; \chi^2_Q = '+ '%3.1f' % chisqQ[jj]+'$'
        plot(xs,ys,color='k',ls=linestyles[jj],label=label)
    l=legend(loc=(.2,.1),numpoints=8)
    l.draw_frame(False)
    xlim(0,300)
    ylim(0,700)
    savefig(plotfilename,format='png')
    
    return 0


def softchisquared(mb,X,Y,yerr,Q):
    """softchisquared: evaluates the logarithm of the objective function
    Input:
       mb=(b,m,Q)   - as in y=mx+b and Q is the soft cut-off
       (actually Q^2 = mb[2]^2+1, e.g., at most 1-sigma clipping)
       X       - independent variable
       Y       - dependent variable
       yerr    - error on the Y
       Q       - Q parameter in chi^2_Q
    History:
       2009-06-01 - Written - Bovy (NYU)
    """
    out= 0.
    for ii in range(len(X)):
        out= out+ 1./(yerr[ii]**2./(Y[ii]-mb[1]*X[ii]-mb[0])**2+m.pow(Q,-2.))
    return out

def ex7b(exclude=sc.array([1,2,3,4]),plotfilename='ex7b.png'):
    """ex7c: solve exercise 7 using an iterative procedure
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
    Output:
       plot
    History:
       2009-06-01 - Written - Bovy (NYU)
    """
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    nsample= ndata- len(exclude)
    #First find the chi-squared solution, which we will use as an
    #initial gues for the bi-exponential optimization
    #Put the dat in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    A= sc.ones((nsample,2))
    C= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
            pass
        else:
            Y[jj]= data[ii][1][1]
            X[jj]= data[ii][1][0]
            A[jj,1]= data[ii][1][0]
            C[jj,jj]= data[ii][2]**2.
            yerr[jj]= data[ii][2]
            jj= jj+1
    #Now compute the best fit and the uncertainties
    bestfit= sc.dot(linalg.inv(C),Y.T)
    bestfit= sc.dot(A.T,bestfit)
    bestfitvar= sc.dot(linalg.inv(C),A)
    bestfitvar= sc.dot(A.T,bestfitvar)
    bestfitvar= linalg.inv(bestfitvar)
    bestfit= sc.dot(bestfitvar,bestfit)
    initialguess= sc.array([bestfit[0],bestfit[1]])
    #With this initial guess start the iteration, using as the weights Q^2/(sigma^2*Q^2+(y-mx-b)^2
    tol= 10**-10.
    Qs= [1.,2.]
    bestfitssoft= sc.zeros((2,len(Qs)))
    chisqQ= sc.zeros(len(Qs))
    for jj in range(len(Qs)):
        currentguess= initialguess
        diff= 2*tol
        while diff > tol:
            oldguess= currentguess
            #Calculate the weight based on the previous iteration
            for ii in range(nsample):
                #Update C
                C[ii,ii]= (yerr[ii]**2.+(Y[ii]-oldguess[1]*X[ii]-oldguess[0])**2/Qs[jj]**2.)
            #Re-fit
            bestfit= sc.dot(linalg.inv(C),Y.T)
            bestfit= sc.dot(A.T,bestfit)
            bestfitvar= sc.dot(linalg.inv(C),A)
            bestfitvar= sc.dot(A.T,bestfitvar)
            bestfitvar= linalg.inv(bestfitvar)
            bestfit= sc.dot(bestfitvar,bestfit)
            currentguess= sc.array([bestfit[0],bestfit[1]])
            diff= m.sqrt((currentguess[0]-oldguess[0])**2/oldguess[0]**2.+(currentguess[1]-oldguess[1])**2/oldguess[1]**2.)       
        bestfitssoft[0,jj]= currentguess[0]
        bestfitssoft[1,jj]= currentguess[1]
        #Calculate chi^2_Q
        for ii in range(nsample):
            chisqQ[jj]= chisqQ[jj]+1./(yerr[ii]**2/(Y[ii]-X[ii]*currentguess[1]-currentguess[1])**2+1./Qs[jj]**2)

    #Now plot the solution
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
    #Plot the best fit line for the different Qs
    linestyles= ('--',':', '-.')
    for jj in range(len(Qs)):
        xlim(0,300)
        ylim(0,700)
        xmin, xmax= xlim()
        nsamples= 1001
        xs= sc.linspace(xmin,xmax,nsamples)
        ys= sc.zeros(nsamples)
        for ii in range(nsamples):
            ys[ii]= bestfitssoft[0,jj]+bestfitssoft[1,jj]*xs[ii]
            if bestfitssoft[0,jj] < 0:
                sgn_str= '-'
            else:
                sgn_str= '+'
        label= r'$Q= '+'%i: y = %4.2f\, x'% (Qs[jj], bestfitssoft[1,jj]) +sgn_str+ '%4.0f ' % m.fabs(bestfitssoft[0,jj])+r'; \chi^2_Q = '+ '%3.1f' % chisqQ[jj]+'$'
        plot(xs,ys,color='k',ls=linestyles[jj],label=label)
    l=legend(loc=(.2,.1),numpoints=8)
    l.draw_frame(False)
    xlim(0,300)
    ylim(0,700)
    savefig(plotfilename,format='png')
    
    return 0


def ex7c(exclude=sc.array([1,2,3,4]),plotfilename='ex7c.png'):
    """ex7d: solve exercise 7 using a simulated annealing optimization
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
    Output:
       plot
    History:
       2009-06-02 - Written - Bovy (NYU)
    """
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    nsample= ndata- len(exclude)
    #First find the chi-squared solution, which we will use as an
    #initial gues for the bi-exponential optimization
    #Put the dat in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    A= sc.ones((nsample,2))
    C= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
            pass
        else:
            Y[jj]= data[ii][1][1]
            X[jj]= data[ii][1][0]
            A[jj,1]= data[ii][1][0]
            C[jj,jj]= data[ii][2]**2.
            yerr[jj]= data[ii][2]
            jj= jj+1
    #Now compute the best fit and the uncertainties
    bestfit= sc.dot(linalg.inv(C),Y.T)
    bestfit= sc.dot(A.T,bestfit)
    bestfitvar= sc.dot(linalg.inv(C),A)
    bestfitvar= sc.dot(A.T,bestfitvar)
    bestfitvar= linalg.inv(bestfitvar)
    bestfit= sc.dot(bestfitvar,bestfit)
    initialguess= sc.array([bestfit[0],bestfit[1]])
    #With this initial guess start off the annealing procedure
    Qs= [1.,2.]
    bestfitssoft= sc.zeros((2,len(Qs)))
    initialchisq= 0.
    for jj in range(nsample):
        initialchisq= initialchisq+(Y[jj]-X[jj]*initialguess[1]-initialguess[0])**2/(yerr[jj]**2)
    chisqQ= sc.zeros(len(Qs))
    for ii in range(len(Qs)):
        chisqQ[ii]= initialchisq
        bestfit= initialguess
        nonglobal= True
        print "Working on Q = "+str(Qs[ii])
        print "Performing 10 runs of the simulating annealing optimization algorithm"
        for jj in range(10):#Do ten runs of the sa algorithm
            sc.random.seed(jj+1) #In the interest of reproducibility (if that's a word)
            bestfitsoft= optimize.anneal(softchisquared,initialguess,(X,Y,yerr,Qs[ii]),
                                     schedule='boltzmann',full_output=1)#,dwell=200,maxiter=1000)
            if bestfitsoft[1] < chisqQ[ii]:
                bestfit= bestfitsoft[0]
                chisqQ[ii]= bestfitsoft[1]
            if bestfitsoft[6] == 0:
                nonglobal= False
        if nonglobal:
            print "Did not cool to the global optimum"
        try:
            x=raw_input('continue to plot? [yn]\n')
        except EOFError:
            print "Since you are in non-interactive mode I will assume 'y'"
            x='y'
        if x == 'n':
            print "returning..."
            return -1
        bestfitssoft[:,ii]= bestfit

    #Now plot the solution
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
    #Plot the best fit line for the different Qs
    linestyles= ('--',':', '-.')
    for jj in range(len(Qs)):
        xlim(0,300)
        ylim(0,700)
        xmin, xmax= xlim()
        nsamples= 1001
        xs= sc.linspace(xmin,xmax,nsamples)
        ys= sc.zeros(nsamples)
        for ii in range(nsamples):
            ys[ii]= bestfitssoft[0,jj]+bestfitssoft[1,jj]*xs[ii]
            if bestfitssoft[0,jj] < 0:
                sgn_str= '-'
            else:
                sgn_str= '+'
        label= r'$Q= '+'%i: y = %4.2f\, x'% (Qs[jj], bestfitssoft[1,jj]) +sgn_str+ '%4.0f ' % m.fabs(bestfitssoft[0,jj])+r'; \chi^2_Q = '+ '%3.1f' % chisqQ[jj]+'$'
        plot(xs,ys,color='k',ls=linestyles[jj],label=label)
    l=legend(loc=(.2,.1),numpoints=8)
    l.draw_frame(False)
    xlim(0,300)
    ylim(0,700)
    savefig(plotfilename,format='png')
    
    return 0


def ex7d(exclude=sc.array([1,2,3,4]),plotfilename='ex7d.png',nburn=100,nsamples=10000,parsigma=[5.,.075]):
    """ex7c: solve exercise 7 using MCMC sampling
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
       nburn          - number of burn-in samples
       nsamples       - number of samples to take after burn-in
       parsigma       - proposal distribution width (Gaussian)
    Output:
       plot
    History:
       2009-06-02 - Written - Bovy (NYU)
    """
    sc.random.seed(-1) #In the interest of reproducibility (if that's a word)
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    nsample= ndata- len(exclude)
    #First find the chi-squared solution, which we will use as an
    #initial gues for the bi-exponential optimization
    #Put the data in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    A= sc.ones((nsample,2))
    C= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
            pass
        else:
            Y[jj]= data[ii][1][1]
            X[jj]= data[ii][1][0]
            A[jj,1]= data[ii][1][0]
            C[jj,jj]= data[ii][2]**2.
            yerr[jj]= data[ii][2]
            jj= jj+1
    #Now compute the best fit and the uncertainties
    bestfit= sc.dot(linalg.inv(C),Y.T)
    bestfit= sc.dot(A.T,bestfit)
    bestfitvar= sc.dot(linalg.inv(C),A)
    bestfitvar= sc.dot(A.T,bestfitvar)
    bestfitvar= linalg.inv(bestfitvar)
    bestfit= sc.dot(bestfitvar,bestfit)
    initialguess= sc.array([bestfit[0],bestfit[1]])
    #With this initial guess start off the sampling procedure
    Qs= [1.,2.]
    bestfitssoft= sc.zeros((2,len(Qs)))
    chisqQ= sc.zeros(len(Qs))
    for kk in range(len(Qs)):
        print "Working on Q = "+str(Qs[kk])
        initialchisqQ= softchisquared(initialguess,X,Y,yerr,Qs[kk])
        bestfit= initialguess
        currentchisqQ= initialchisqQ
        bestchisqQ= initialchisqQ
        currentguess= initialguess
        naccept= 0
        for jj in range(nburn+nsamples):
            #Draw a sample from the proposal distribution
            newsample= sc.zeros(2)
            newsample[0]= currentguess[0]+stats.norm.rvs()*parsigma[0]
            newsample[1]= currentguess[1]+stats.norm.rvs()*parsigma[1]
            #Calculate the objective function for the newsample
            newchisqQ= softchisquared(newsample,X,Y,yerr,Qs[kk])
            #Accept or reject
            #Reject with the appropriate probability
            u= stats.uniform.rvs()
            if u < m.exp(currentchisqQ-newchisqQ):
                #Accept
                currentchisqQ= newchisqQ
                currentguess= newsample
                naccept= naccept+1
            if currentchisqQ < bestchisqQ:
                bestfit= currentguess
                bestchisqQ= currentchisqQ
        bestfitssoft[:,kk]= bestfit
        chisqQ[kk]= bestchisqQ
        if double(naccept)/(nburn+nsamples) < .5 or double(naccept)/(nburn+nsamples) > .8:
            print "Acceptance ratio was "+str(double(naccept)/(nburn+nsamples))

    #Now plot the solution
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
    #Plot the best fit line for the different Qs
    linestyles= ('--',':', '-.')
    for jj in range(len(Qs)):
        xlim(0,300)
        ylim(0,700)
        xmin, xmax= xlim()
        nsamples= 1001
        xs= sc.linspace(xmin,xmax,nsamples)
        ys= sc.zeros(nsamples)
        for ii in range(nsamples):
            ys[ii]= bestfitssoft[0,jj]+bestfitssoft[1,jj]*xs[ii]
            if bestfitssoft[0,jj] < 0:
                sgn_str= '-'
            else:
                sgn_str= '+'
        label= r'$Q= '+'%i: y = %4.2f\, x'% (Qs[jj], bestfitssoft[1,jj]) +sgn_str+ '%4.0f ' % m.fabs(bestfitssoft[0,jj])+r'; \chi^2_Q = '+ '%3.1f' % chisqQ[jj]+'$'
        plot(xs,ys,color='k',ls=linestyles[jj],label=label)
    l=legend(loc=(.2,.1),numpoints=8)
    l.draw_frame(False)
    xlim(0,300)
    ylim(0,700)
    savefig(plotfilename,format='png')
    
    return 0

