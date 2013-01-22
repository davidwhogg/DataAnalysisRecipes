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
import scipy.stats as stats
import scipy.linalg as linalg
import math as m

def sample_normal(mean,covar,nsamples=1):
    """sample_normal: Sample a d-dimensional Gaussian distribution with
    mean and covar.

    Input:
       mean     - the mean of the Gaussian
       covar    - the covariance of the Gaussian
       nsamples - (optional) the number of samples desired

    Output:
       samples; if nsamples != 1 then a list is returned

    Dependencies:
       scipy
       scipy.stats.norm
       scipy.linalg.cholesky

    History:
       2009-05-20 - Written - Bovy (NYU)
    """
    p= covar.shape[0]
    #First lower Cholesky of covar
    L= linalg.cholesky(covar,lower=True)
    if nsamples > 1:
        out= []
    for kk in range(nsamples):
        #Generate a vector in which the elements ~N(0,1)
        y= sc.zeros(p)
        for ii in range(p):
            y[ii]= stats.norm.rvs()
        #Form the sample as Ly+mean
        thissample= sc.dot(L,y)+mean
        if nsamples == 1:
            return thissample
        else:
            out.append(thissample)
    return out


