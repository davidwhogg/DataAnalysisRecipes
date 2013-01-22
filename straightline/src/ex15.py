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

def ex15(exclude=sc.array([1,2,3,4]),plotfilename='ex15.png',
		 bovyprintargs={}):
    """ex15: solve exercise 15
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
    #Put the dat in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    Z= sc.zeros((nsample,2))
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
            pass
        else:
            Y[jj]= data[ii][1][1]
            X[jj]= data[ii][1][0]
            Z[jj,0]= X[jj]
            Z[jj,1]= Y[jj]
            jj= jj+1
    #Now compute the PCA solution
    Zm= sc.mean(Z,axis=0)
    Q= sc.cov(Z.T)
    eigs= linalg.eig(Q)
    maxindx= sc.argmax(eigs[0])
    V= eigs[1][maxindx]
    V= V/linalg.norm(V)

    m= sc.sqrt(1/V[0]**2.-1)
    bestfit= sc.array([-m*Zm[0]+Zm[1],m])

    #Plot result
    plot.bovy_print(**bovyprintargs)
    xrange=[0,300]
    yrange=[0,700]
    plot.bovy_plot(sc.array(xrange),bestfit[1]*sc.array(xrange)+bestfit[0],
                   'k--',xrange=xrange,yrange=yrange,
                   xlabel=r'$x$',ylabel=r'$y$',zorder=2)
    plot.bovy_plot(X,Y,marker='o',color='k',linestyle='None',
                   zorder=0,overplot=True)
 
    plot.bovy_text(r'$y = %4.2f \,x %4.0f' % (bestfit[1], bestfit[0])+r'$',
                   bottom_right=True)
    plot.bovy_end_print(plotfilename)



def objective(mb,Z,ycovar):
    """The objective function"""
    v= 1./sc.sqrt(1+mb[1]**2.)*sc.array([-mb[1],1.])
    cost= v[1]
    delta= sc.dot(v,Z.T)-mb[0]*cost
    sigma2= sc.dot(v,sc.dot(ycovar,v))
    return 0.5*sc.sum(delta**2./sigma2)

