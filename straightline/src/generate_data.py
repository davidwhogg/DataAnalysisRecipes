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
import re
import scipy as sc
import scipy.stats as stats
import scipy.linalg as linalg
import math as m
import numpy as nu
from sample_wishart import sample_wishart
from sample_normal import sample_normal

def generate_data(ndata=20,nback=4,yerr=.05,errprop=2,wishartshape=5):
    """generate_data: Generate the data that is to be fit with a straight line

    Input:
       ndata    - Total number of data points to generate
       nback    - Number of data points to generate from the background
       yerr     - typical fractional y error
       errprop  - proportionality constant between typical y and typical
                  x error
       wishartshape - shape parameter for the Wishart density from which
                      the error covariances are drawn

    Output:
       list of { array(data point), array(errorcovar) }
    
    History:
       2009-05-20 - Started - Bovy (NYU)
    """
    nu.random.seed(8) #In the interest of reproducibility (if that's a word)
    #The distribution underlying the straight line is a Gaussian, with a large
    #eigenvalue in the direction of the line and a small eigenvalue in the
    #direction orthogonal to this
    #Draw a random slope (by drawing a random angle such that tan angle = slope
    alpha= stats.uniform.rvs()*m.pi-m.pi/2.
    slope= m.tan(alpha)
    #Draw a random intercept from intercept ~ 1/intercept intercept \in [.1,10]
    intercept= stats.uniform.rvs()*2.-1.
    intercept= 10.**intercept
    #print slope, intercept
    rangey= intercept*10.
    rangeline= rangey/m.sin(alpha)
    rangex= rangey/slope
    #Now draw the variances of the underlying Gaussian
    #We want one to be big
    multiplerangeline= 1.
    multiplerange2= 10.
    sigma1= nu.random.gamma(2,.5/(multiplerangeline*rangeline)**2)
    sigma1= 1/m.sqrt(sigma1)
    #And the other one the be small
    sigma2= nu.random.gamma(2,.5*(multiplerange2*rangeline)**2)
    sigma2= 1/m.sqrt(sigma2)
    covar= sc.array([[sigma1**2,0.],[0.,sigma2**2]])
    #Rotate the covariance matrix
    rotationmatrix= sc.array([[m.cos(alpha),-m.sin(alpha)],
                              [m.sin(alpha),m.cos(alpha)]])
    modelcovar= sc.dot(rotationmatrix,covar)
    modelcovar=sc.dot(modelcovar,rotationmatrix.transpose())
    #Also set the mean
    modelmean= sc.array([0.,intercept+5*rangey])
    modelmean[0]= (modelmean[1]-intercept)/slope
    #The background covar
    backcovar= sc.array([[4*rangex**2.,0.],[0.,4*rangey**2.]])
    #Now start drawing samples from this
    out=[]
    for ii in range(ndata):
        #First set-up an error covariance. Use the fractional error to
        #multiply the ymean, use the proportionality between yerr and xerr
        #to get the error in x, and draw a random angle for the correlation
        #But not allow for completely correlated erors
        #Draw a random error covariance from an inverse Wishart
        #distribution that has the constructed error covariance as its' center'
        correlation_angle= stats.uniform.rvs()*m.pi/2+m.pi/4
        thisyerr= (yerr*modelmean[1])**2.
        thisxerr= thisyerr/errprop/slope**2.
        thiscorrelation= m.cos(correlation_angle)
        thiscovxy= thiscorrelation*m.sqrt(thisxerr*thisyerr)
        thissampleerr= sc.array([[thisxerr,thiscovxy],[thiscovxy,thisyerr]])
        sampleerr= sample_wishart(
            wishartshape,linalg.inv(thissampleerr)/wishartshape)
        sampleerr= linalg.inv(sampleerr)
        #Now draw a sample from the model distribution convolved with this
        #error distribution
        if ii < nback:
            samplethiscovar= sampleerr+backcovar
        else:
            samplethiscovar= sampleerr+modelcovar
        thissample= sample_normal(modelmean,samplethiscovar)
        sample=[]
        sample.append(thissample)
        sample.append(sampleerr)
        out.append(sample)
        
    return out

def sign(x):
    if x < 0: return -1
    else: return 1
    
def write_table_to_file(filename,latex=False,allerr=False,ndec=[0,0,0,0,2]):
    """write_table_to_file: Write the generated data to a latex table
    Includes {x_i,y_i,sigma_yi}

    Input:
       filename  - filename for table
       latex     - Write latex file
       allerr    - If True, write all of the errors
       ndec      - number of decimal places (array with five members)

    History:
       2009-05-20 - Started - Bovy (NYU)
    """
    #First generate the data
    data= generate_data()
    #Set up the file
    outfile=open(filename,'w')
    if allerr:
        ncol= 5
    else:
        ncol= 3
    #First write all of the table header
    nextra= 0
    if latex:
        outfile.write(r'\begin{deluxetable}{')
        outfile.write('r')
        for jj in range(ncol):
            outfile.write('r')
            if ndec[jj] != 0:
                nextra= nextra+1
                outfile.write(r'@{.}l')
        outfile.write('}\n')
        ntablecols= ncol+nextra+1
        outfile.write(r'\tablecolumns{'+str(ntablecols)+'}'+'\n')
        outfile.write(r'\tablehead{ID &')
        #x
        if ndec[0] != 0:
            outfile.write(r'\multicolumn{2}{c}{$x$} & ')
        else:
            outfile.write(r'$x$ & ')
        #y
        if ndec[1] != 0:
            outfile.write(r'\multicolumn{2}{c}{$y$} & ')
        else:
            outfile.write(r'$y$ & ')
        #sigma_y
        if ndec[2] != 0:
            outfile.write(r'\multicolumn{2}{c}{$\sigma_y$}')
        else:
            outfile.write(r'$\sigma_y$')
        if allerr:
            #sigma_x
            if ndec[3] != 0:
                outfile.write(r' & \multicolumn{2}{c}{$\sigma_x$} & ')
            else:
                outfile.write(r' & $\sigma_x$ & ')
            #rho_{xy}
            if ndec[4] != 0:
                outfile.write(r' \multicolumn{2}{c}{$\rho_{xy}$}')
            else:
                outfile.write(r' $\rho_{xy}')
        outfile.write(r'}'+'\n')
        outfile.write(r'\tablewidth{0pt}'+'\n')
        outfile.write(r'\startdata'+'\n')
    else:
        if allerr:
            outfile.write('#Data from Table 2\n')
            outfile.write('#ID\tx\ty\t\sigma_y\t\sigma_x\t'+r'\rho_{xy}'+'\n')
        else:
            outfile.write('#Data from Table 1\n')
            outfile.write('#ID\tx\ty\t\sigma_y\n')
    #Then write the data
    for ii in range(len(data)):
        #Write the ID
        if latex:
            outfile.write(str(ii+1)+' & ')
        else:
            outfile.write(str(ii+1)+'\t')
        #Write x and y
        for jj in range(2):
            if sign(data[ii][0][jj]) == -1:
                sign_str= '-'
            else:
                sign_str= ''
            int_part=abs(long(data[ii][0][jj]))
            dec_part= long(round(10**ndec[jj]*abs(data[ii][0][jj]-long(data[ii][0][jj]))))
            if dec_part >= 10**ndec[jj]:
                int_part = int_part+1
                dec_part= dec_part-10**ndec[jj]
            int_part= str(int_part)
            if dec_part == 0:
                sign_str=''
            dec_part='%i' % dec_part
            dec_part= dec_part.zfill(ndec[jj])
            if latex:
                if ndec[jj] != 0:
                    outfile.write(sign_str+int_part+' & '+dec_part + ' & ')
                else:
                    outfile.write(sign_str+int_part+' & ')
            else:
                if ndec[jj] != 0:
                    outfile.write(sign_str+int_part+'.'+dec_part+'\t')
                else:
                    outfile.write(sign_str+int_part+'\t')
        #Write sigma_y
        sigma_y= m.sqrt(data[ii][1][1,1])
        if sign(sigma_y) == -1:
            sign_str= '-'
        else:
            sign_str= ''
        int_part=abs(long(sigma_y))
        dec_part= long(round(10**ndec[2]*abs(sigma_y-long(sigma_y))))
        if dec_part >= 10**ndec[2]:
            int_part = int_part+1
            dec_part= dec_part-10**ndec[2]
        int_part= str(int_part)
        if dec_part == 0:
            sign_str=''
        dec_part='%i' % dec_part
        dec_part= dec_part.zfill(ndec[2])
        if latex:
            if ndec[2] != 0:
                outfile.write(sign_str+int_part+' & '+dec_part)
            else:
                outfile.write(sign_str+int_part)
        else:
            if ndec[2] != 0:
                outfile.write(sign_str+int_part+'.'+dec_part)
            else:
                outfile.write(sign_str+int_part)
        if allerr:
            #Write sigma_x
            sigma_x= m.sqrt(data[ii][1][0,0])
            if sign(sigma_x) == -1:
                sign_str= '-'
            else:
                sign_str= ''
            int_part=abs(long(sigma_x))
            dec_part= long(round(10**ndec[3]*abs(sigma_x-long(sigma_x))))
            if dec_part >= 10**ndec[3]:
                int_part = int_part+1
                dec_part= dec_part-10**ndec[3]
            int_part= str(int_part)
            if dec_part == 0:
                sign_str=''
            dec_part='%i' % dec_part
            dec_part= dec_part.zfill(ndec[3])
            if latex:
                if ndec[3] != 0:
                    outfile.write(' & '+sign_str+int_part+' & '+dec_part +' & ')
                else:
                    outfile.write(' & '+sign_str+int_part + ' & ')
            else:
                if ndec[3] != 0:
                    outfile.write('\t'+sign_str+int_part+'.'+dec_part+'\t')
                else:
                    outfile.write('\t'+sign_str+int_part+'\t')
            #Write rho_{xy}
            rho_xy= data[ii][1][0,1]/sigma_x/sigma_y
            if sign(rho_xy) == -1:
                sign_str= '-'
            else:
                sign_str= ''
            int_part=abs(long(rho_xy))
            dec_part= long(round(10**ndec[4]*abs(rho_xy-long(rho_xy))))
            if dec_part >= 10**ndec[4]:
                int_part = int_part+1
                dec_part= dec_part-10**ndec[4]
            int_part= str(int_part)
            if dec_part == 0:
                sign_str=''
            dec_part='%i' % dec_part
            dec_part= dec_part.zfill(ndec[4])
            if latex:
                if ndec[4] != 0:
                    outfile.write(sign_str+int_part+' & '+dec_part)
                else:
                    outfile.write(sign_str+int_part)
            else:
                if ndec[4] != 0:
                    outfile.write(sign_str+int_part+'.'+dec_part)
                else:
                    outfile.write(sign_str+int_part)
            
        if latex:
            outfile.write(r'\\'+'\n')
        else:
            outfile.write('\n')
    #Write the footer
    if latex:
        if allerr:
            outfile.write(r'\tablecomments{The full uncertainty covariance matrix for each data point is given by\\ $\left[\begin{array}{cc} \sigma_x^2 & \rho_{xy}\sigma_x\sigma_y\\\rho_{xy}\sigma_x\sigma_y & \sigma_y^2\end{array}\right]$.}'+'\n')
            outfile.write(r'\label{table:data_allerr}'+'\n')
        else:
            outfile.write(r'\tablecomments{$\sigma_y$ is the uncertainty for the $y$ measurement.}'+'\n')
            outfile.write(r'\label{table:data_yerr}'+'\n')
        outfile.write(r'\enddata'+'\n')
        outfile.write(r'\end{deluxetable}'+'\n')
    outfile.close()
    
    return 0

def read_data(datafilename='data_yerr.dat',allerr=False):
    """read_data_yerr: Read the data from the file into a python structure
    Reads {x_i,y_i,sigma_yi}

    Input:
       datafilename    - the name of the file holding the data
       allerr          - If set to True, read all of the errors

    Output:
       Returns a list {i,datapoint, y_err}, or {i,datapoint,y_err, x_err, corr}

    History:
       2009-05-20 - Started - Bovy (NYU)
    """
    if allerr:
        ncol= 6
    else:
        ncol= 4
    #Open data file
    datafile= open(datafilename,'r')
    #catch-all re that reads numbers
    expr= re.compile(r"-?[0-9]+(\.[0-9]*)?(E\+?-?[0-9]+)?")
    rawdata=[]
    nline= 0
    for line in datafile:
        if line[0] == '#':#Comments
            continue
        nline+= 1
        values= expr.finditer(line)
        nvalue= 0
        for i in values:
            rawdata.append(float(i.group()))
            nvalue+= 1
        if nvalue != ncol:
            print("Warning, number of columns for this record does not match the expected number")
    #Now process the raw data
    out=[]
    for ii in range(nline):
        #First column is the data number
        thissample= []
        thissample.append(rawdata[ii*ncol])
        sample= sc.array([rawdata[ii*ncol+1],rawdata[ii*ncol+2]])
        thissample.append(sample)
        thissample.append(rawdata[ii*ncol+3])
        if allerr:
            thissample.append(rawdata[ii*ncol+4])
            thissample.append(rawdata[ii*ncol+5])
        out.append(thissample)
    return out
