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
import scipy.linalg as linalg
import math as m
from generate_data import read_data
import bovy_plot as plot

def ex9(exclude=sc.array([1,2,3,4]),plotfilename='ex9.png',zoom=False,
		bovyprintargs={}):
    """ex9: solve exercise 9

    Input:
       exclude  - ID numbers to exclude from the analysis
       zoom - zoom in
    Output:
       plot
    History:
       2009-05-27 - Written - Bovy (NYU)
    """
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    nsample= ndata- len(exclude)
    nSs= 1001
    if zoom:
        Srange=[900,1000]
    else:
        Srange=[0.001,1500]
    Ss= sc.linspace(Srange[0],Srange[1],nSs)
    chi2s= sc.zeros(nSs)
    for kk in range(nSs):
        #Put the dat in the appropriate arrays and matrices
        Y= sc.zeros(nsample)
        A= sc.ones((nsample,2))
        C= sc.zeros((nsample,nsample))
        yerr= sc.zeros(nsample)
        jj= 0
        for ii in range(ndata):
            if sc.any(exclude == data[ii][0]):
                pass
            else:
                Y[jj]= data[ii][1][1]
                A[jj,1]= data[ii][1][0]
                C[jj,jj]= Ss[kk]
                yerr[jj]= data[ii][2]#OMG, such bad code
                jj= jj+1
        #Now compute the best fit and the uncertainties
        bestfit= sc.dot(linalg.inv(C),Y.T)
        bestfit= sc.dot(A.T,bestfit)
        bestfitvar= sc.dot(linalg.inv(C),A)
        bestfitvar= sc.dot(A.T,bestfitvar)
        bestfitvar= linalg.inv(bestfitvar)
        bestfit= sc.dot(bestfitvar,bestfit)
        chi2s[kk]= chi2(bestfit,A,Y,C)

    #Now plot the solution
    plot.bovy_print(**bovyprintargs)
    #Plot the best fit line
    xrange=Srange
    if zoom:
        yrange=[nsample-4,nsample]
    else:
        yrange=[nsample-10,nsample+8]
    plot.bovy_plot(Ss,
                   chi2s,
                   'k-',xrange=xrange,yrange=yrange,
                   xlabel=r'$S$',ylabel=r'$\chi^2$',zorder=1)
    plot.bovy_plot(sc.array(Srange),sc.array([nsample-2,nsample-2]),
                   'k--',zorder=2,overplot=True)
    #plot.bovy_plot(sc.array([sc.median(yerr**2.),sc.median(yerr**2.)]),
    #               sc.array(yrange),color='0.75',overplot=True)
    plot.bovy_plot(sc.array([sc.mean(yerr**2.),sc.mean(yerr**2.)]),
                   sc.array(yrange),color='0.75',overplot=True)
    plot.bovy_end_print(plotfilename)

    return 0

def chi2(X,A,Y,C):
    """return chi2"""
    yminusax= Y-sc.dot(A,X)
    return sc.dot(yminusax,sc.dot(linalg.inv(C),yminusax))
    
