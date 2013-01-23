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
import numpy

from ex1 import ex1
from ex3 import ex3
from exMix1 import exMix1
from ex9 import ex9
from ex10 import ex10
from ex12 import ex12
from ex13 import ex13
from ex14 import ex14
from ex15 import ex15
from ex16 import ex16
from ex17 import ex17

from astrometry.util.file import *

if __name__ == '__main__':

	if False:
		from matplotlib.cm import register_cmap
		_antigray_data =  {'red':   ((0., 1, 1), (1., 0, 0)),
						   'green': ((0., 1, 1), (1., 0, 0)),
						   'blue':  ((0., 1, 1), (1., 0, 0))}
		register_cmap(name='gist_yarg',
					  data=_antigray_data)


	bpa = dict(axes_labelsize=14, text_fontsize=12)

	ex1(plotfilename='ex1.pdf', bovyprintargs=bpa)
	ex1(exclude=[], plotfilename='ex2.pdf', bovyprintargs=bpa)

	ex3(plotfilename='ex3.pdf', bovyprintargs=bpa)

	sd1 = None
	pfn1 = 'sd1.pickle'
	if os.path.exists(pfn1):
		print 'Reading sample data from', pfn1
		sd1 = unpickle_from_file(pfn1)
	sd2 = None
	pfn2 = 'sd2.pickle'
	if os.path.exists(pfn2):
		print 'Reading sample data from', pfn2
		sd2 = unpickle_from_file(pfn2)

	sd1 = exMix1(plotfilenameA='exMix1a.pdf',
				 plotfilenameB='exMix1b.pdf',
				 plotfilenameC='exMix1c.pdf',
				 bovyprintargs=bpa, sampledata=sd1) #, nburn=10000, nsamples=10000)
	sd2 = exMix1(dsigma=2,
				 plotfilenameA='exMix2a.pdf',
				 plotfilenameB='exMix2b.pdf',
				 plotfilenameC='exMix2c.pdf',
				 bovyprintargs=bpa, sampledata=sd2) #, nburn=10000, nsamples=10000)

	if not os.path.exists(pfn1):
		print 'Saving sample data to', pfn1
		pickle_to_file(sd1, pfn1)
	if not os.path.exists(pfn2):
		print 'Saving sample data to', pfn2
		pickle_to_file(sd2, pfn2)




	exclude=numpy.array([1,2,3,4])

	sd3 = None
	pfn3 = 'sd3.pickle'
	if os.path.exists(pfn3):
		print 'Reading sample data from', pfn3
		sd3 = unpickle_from_file(pfn3)
	# from exNew.py
	parsigma=[5,.075,.01,1,.1]

	sd3 = exMix1(exclude=exclude,
				 parsigma=parsigma,
				 plotfilenameA='exMix3a.pdf',
				 plotfilenameB='exMix3b.pdf',
				 plotfilenameC='exMix3c.pdf',
				 bovyprintargs=bpa, sampledata=sd3)#, nburn=10000, nsamples=100000)
	if not os.path.exists(pfn3):
		print 'Saving sample data to', pfn3
		pickle_to_file(sd3, pfn3)


	sd4 = None
	pfn4 = 'sd4.pickle'
	if os.path.exists(pfn4):
		print 'Reading sample data from', pfn4
		sd4 = unpickle_from_file(pfn4)

	sd4 = exMix1(exclude=exclude,
				 parsigma=parsigma,
				 dsigma=2,
				 plotfilenameA='exMix4a.pdf',
				 plotfilenameB='exMix4b.pdf',
				 plotfilenameC='exMix4c.pdf',
				 bovyprintargs=bpa, sampledata=sd4)#, nburn=10000, nsamples=100000)

	if not os.path.exists(pfn4):
		print 'Saving sample data to', pfn4
		pickle_to_file(sd4, pfn4)

	print 'ex9'
	ex9(plotfilename='ex9a.pdf', bovyprintargs=bpa)
	ex9(plotfilename='ex9b.pdf', zoom=True, bovyprintargs=bpa)
	
	print 'ex10'
	ex10(plotfilenameA='ex10a.pdf', plotfilenameB='ex10b.pdf',
		 bovyprintargs=bpa)

	print 'ex12'
	ex12(plotfilename='ex12.pdf', bovyprintargs=bpa)

	print 'ex13'
	ex12(exclude=[], plotfilename='ex13a.pdf', bovyprintargs=bpa)
	ex13(exclude=[], plotfilename='ex13b.pdf', bovyprintargs=bpa)

	print 'ex14'
	ex14(plotfilename='ex14.pdf', bovyprintargs=bpa)
	
	print 'ex15'
	ex15(plotfilename='ex15.pdf', bovyprintargs=bpa)

	print 'ex16'
	ex16(plotfilename='ex16.pdf', bovyprintargs=bpa)

	print 'ex17'
	ex17(plotfilename='ex17.pdf', bovyprintargs=bpa)
