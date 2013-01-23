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


def ex14(exclude=sc.array([1,2,3,4]),plotfilename='ex14.png',
		 bovyprintargs={}):
    """ex12: solve exercise 14
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
    Output:
       plot
    History:
       2010-05-07 - Written - Bovy (NYU)
    """
    #Read the data
    data= read_data('data_allerr.dat',allerr=True)
    ndata= len(data)
    nsample= ndata- len(exclude)
    #First find the chi-squared solution, which we will use as an
    #initial gues
    #Put the dat in the appropriate arrays and matrices
    Y1= sc.zeros(nsample)
    X1= sc.zeros(nsample)
    A1= sc.ones((nsample,2))
    C1= sc.zeros((nsample,nsample))
    Y2= sc.zeros(nsample)
    X2= sc.zeros(nsample)
    A2= sc.ones((nsample,2))
    C2= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    xerr= sc.zeros(nsample)
    ycovar= sc.zeros((2,nsample,2))#Makes the sc.dot easier
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
            pass
        else:
            Y1[jj]= data[ii][1][1]
            X1[jj]= data[ii][1][0]
            A1[jj,1]= data[ii][1][0]
            C1[jj,jj]= data[ii][2]**2.
            yerr[jj]= data[ii][2]
            Y2[jj]= data[ii][1][0]
            X2[jj]= data[ii][1][1]
            A2[jj,1]= data[ii][1][1]
            C2[jj,jj]= data[ii][3]**2.
            xerr[jj]= data[ii][3]
            jj= jj+1
    #Now compute the best fit and the uncertainties: forward
    bestfit1= sc.dot(linalg.inv(C1),Y1.T)
    bestfit1= sc.dot(A1.T,bestfit1)
    bestfitvar1= sc.dot(linalg.inv(C1),A1)
    bestfitvar1= sc.dot(A1.T,bestfitvar1)
    bestfitvar1= linalg.inv(bestfitvar1)
    bestfit1= sc.dot(bestfitvar1,bestfit1)
    #Now compute the best fit and the uncertainties: backward
    bestfit2= sc.dot(linalg.inv(C2),Y2.T)
    bestfit2= sc.dot(A2.T,bestfit2)
    bestfitvar2= sc.dot(linalg.inv(C2),A2)
    bestfitvar2= sc.dot(A2.T,bestfitvar2)
    bestfitvar2= linalg.inv(bestfitvar2)
    bestfit2= sc.dot(bestfitvar2,bestfit2)
    #Propagate to y=mx+b
    linerrprop= sc.array([[-1./bestfit2[1],bestfit2[0]/bestfit2[1]**2],
                          [0.,-1./bestfit2[1]**2.]])
    bestfit2= sc.array([-bestfit2[0]/bestfit2[1],1./bestfit2[1]])
    bestfitvar2= sc.dot(linerrprop,sc.dot(bestfitvar2,linerrprop.T))

    #Plot result
    plot.bovy_print(**bovyprintargs)
    xrange=[0,300]
    yrange=[0,700]
    plot.bovy_plot(sc.array(xrange),bestfit1[1]*sc.array(xrange)+bestfit1[0],
                   'k--',xrange=xrange,yrange=yrange,
                   xlabel=r'$x$',ylabel=r'$y$',zorder=2)
    plot.bovy_plot(sc.array(xrange),bestfit2[1]*sc.array(xrange)+bestfit2[0],
                   'k-.',overplot=True,zorder=2)

    #Plot data
    errorbar(A1[:,1],Y1,yerr,xerr,color='k',marker='o',
             linestyle='None',zorder=0)
    plot.bovy_text(r'$\mathrm{forward}\ ---\:\ y = ( '+'%4.2f \pm %4.2f )\,x+ ( %4.0f\pm %4.0f' % (bestfit1[1], m.sqrt(bestfitvar1[1,1]), bestfit1[0],m.sqrt(bestfitvar1[0,0]))+r')$'+'\n'+
                   r'$\mathrm{reverse}\ -\cdot -\:\ y = ( '+'%4.2f \pm %4.2f )\,x+ ( %4.0f\pm %4.0f' % (bestfit2[1], m.sqrt(bestfitvar2[1,1]), bestfit2[0],m.sqrt(bestfitvar2[0,0]))+r')$',bottom_right=True)
    plot.bovy_end_print(plotfilename)



def objective(mb,Z,ycovar):
    """The objective function"""
    v= 1./sc.sqrt(1+mb[1]**2.)*sc.array([-mb[1],1.])
    cost= v[1]
    delta= sc.dot(v,Z.T)-mb[0]*cost
    sigma2= sc.dot(v,sc.dot(ycovar,v))
    return 0.5*sc.sum(delta**2./sigma2)

