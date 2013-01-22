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
import matplotlib
matplotlib.use('Agg')
from pylab import *
from matplotlib.pyplot import *
from matplotlib import rc
from matplotlib.pyplot import title as pytitle
from matplotlib.patches import Ellipse

def plot_data_yerr():
    """plot_data_yerr: Plot the data with the error bars in the y-direction

    History:
       2009-05-20 - Written - Bovy (NYU)

    """
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    #Put the data into x, y, and yerr
    id= sc.zeros(ndata)
    x= sc.zeros(ndata)
    y= sc.zeros(ndata)
    yerr= sc.zeros(ndata)
    for ii in range(ndata):
        id[ii]= data[ii][0]
        x[ii]= data[ii][1][0]
        y[ii]= data[ii][1][1]
        yerr[ii]= data[ii][2]
        
    plotfilename='data_yerr.png'
    fig_width=7.5
    fig_height=7.5
    fig_size =  [fig_width,fig_height]
    params = {'axes.labelsize': 12,
              'text.fontsize': 11,
              'legend.fontsize': 12,
              'xtick.labelsize':10,
              'ytick.labelsize':10,
              'text.usetex': True,
              'figure.figsize': fig_size}
    rcParams.update(params)
    errorbar(x,y,yerr,marker='o',color='k',linestyle='None')
    xlabel(r'$x$')
    ylabel(r'$y$')
    savefig(plotfilename,format='png')

    return 0

def plot_data_allerr():
    """plot_data_allerr: Plot the data with full error ellipses

    History:
       2009-05-20 - Written - Bovy (NYU)

    """
    #Read the data
    data= read_data('data_allerr.dat',True)
    ndata= len(data)
    #Create the ellipses and the data points
    id= sc.zeros(ndata)
    x= sc.zeros(ndata)
    y= sc.zeros(ndata)
    ellipses=[]
    ymin, ymax= 0, 0
    xmin, xmax= 0,0
    for ii in range(ndata):
        id[ii]= data[ii][0]
        x[ii]= data[ii][1][0]
        y[ii]= data[ii][1][1]
        #Calculate the eigenvalues and the rotation angle
        ycovar= sc.zeros((2,2))
        ycovar[0,0]= data[ii][3]**2.
        ycovar[1,1]= data[ii][2]**2.
        ycovar[0,1]= data[ii][4]*m.sqrt(ycovar[0,0]*ycovar[1,1])
        ycovar[1,0]= ycovar[0,1]
        eigs= linalg.eig(ycovar)
        angle= m.atan(-eigs[1][0,1]/eigs[1][1,1])/m.pi*180.
        #print x[ii], y[ii], m.sqrt(ycovar[1,1]), m.sqrt(ycovar[0,0])
        #print m.sqrt(eigs[0][0]), m.sqrt(eigs[0][1]), angle
        thisellipse= Ellipse(sc.array([x[ii],y[ii]]),2*m.sqrt(eigs[0][0]),
                             2*m.sqrt(eigs[0][1]),angle)
        ellipses.append(thisellipse)
        if (x[ii]+m.sqrt(ycovar[0,0])) > xmax:
            xmax= (x[ii]+m.sqrt(ycovar[0,0]))
        if (x[ii]-m.sqrt(ycovar[0,0])) < xmin:
            xmin= (x[ii]-m.sqrt(ycovar[0,0]))
        if (y[ii]+m.sqrt(ycovar[1,1])) > ymax:
            ymax= (y[ii]+m.sqrt(ycovar[1,1]))
        if (y[ii]-m.sqrt(ycovar[1,1])) < ymin:
            ymin= (y[ii]-m.sqrt(ycovar[1,1]))
        
    plotfilename='data_allerr.png'
    fig_width=7.5
    fig_height=7.5
    fig_size =  [fig_width,fig_height]
    params = {'axes.labelsize': 12,
              'text.fontsize': 11,
              'legend.fontsize': 12,
              'xtick.labelsize':10,
              'ytick.labelsize':10,
              'text.usetex': True,
              'figure.figsize': fig_size}
    rcParams.update(params)
    fig= figure()
    ax= fig.add_subplot(111)
    #Add the error ellipses
    for e in ellipses:
        ax.add_artist(e)
        e.set_facecolor('none')
    ax.plot(x,y,color='k',marker='o',linestyle='None')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    savefig(plotfilename,format='png')

    return 0

