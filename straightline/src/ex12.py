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


def ex12(exclude=sc.array([1,2,3,4]),plotfilename='ex12.png',
		 bovyprintargs={}):
    """ex12: solve exercise 12 by optimization of the objective function
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
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
    #Now optimize
    bestfit2d1= optimize.fmin(objective,bestfit,(Z,ycovar),disp=False)
    #Restart the optimization once using a different method
    bestfit2d= optimize.fmin_powell(objective,bestfit,
                                       (Z,ycovar),disp=False)
    if linalg.norm(bestfit2d-bestfit2d1) > 10**-12:
        if linalg.norm(bestfit2d-bestfit2d1) < 10**-6:
            print("Different optimizers give slightly different results...")
        else:
            print("Different optimizers give rather different results...")
        print("The norm of the results differs by %g" % linalg.norm(bestfit2d-bestfit2d1))
        try:
            x=raw_input('continue to plot? [yn]\n')
        except EOFError:
            print("Since you are in non-interactive mode I will assume 'y'")
            x='y'
        if x == 'n':
            print("returning...")
            return -1

    #Plot result
    plot.bovy_print(**bovyprintargs)
    xrange=[0,300]
    yrange=[0,700]
    plot.bovy_plot(sc.array(xrange),bestfit2d[1]*sc.array(xrange)+bestfit2d[0],
                   'k-',xrange=xrange,yrange=yrange,
                   xlabel=r'$x$',ylabel=r'$y$',zorder=2)
     
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

    plot.bovy_text(r'$y = %4.2f \,x+ %4.0f' % (bestfit2d[1], bestfit2d[0])+r'$',
                   bottom_right=True)
    plot.bovy_end_print(plotfilename)



def objective(mb,Z,ycovar):
    """The objective function"""
    v= 1./sc.sqrt(1+mb[1]**2.)*sc.array([-mb[1],1.])
    cost= v[1]
    delta= sc.dot(v,Z.T)-mb[0]*cost
    sigma2= sc.dot(v,sc.dot(ycovar,v))
    return 0.5*sc.sum(delta**2./sigma2+sc.log(sigma2)+sc.log(1.+mb[1]**2.))

