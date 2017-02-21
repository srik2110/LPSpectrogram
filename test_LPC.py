import matplotlib.pyplot as plt 
import pylab
from scipy.io import wavfile
from scipy.linalg import toeplitz
from scipy import signal
import numpy as np

pylab.close('all')
#w1 = wave.open('/home/srik/MLSP/speechFiles/clean.wav')
#w2 = wave.open('/home/srik/MLSP/speechFiles/noise.wav')
fs, w1 = wavfile.read('/home/srik/MLSP/speechFiles/clean.wav')

tWindow = 25e-3#Window length in time
NWindow = int(tWindow*fs)#no. of elements in window
window = np.hamming(NWindow)#window type
NOverlap = int(0.6*NWindow)#60% overlap

#take every frame and compute autocorrelation
order = 399
nframes = 311
lincoeff = np.empty((order-1, 0))
psLog = np.empty((201, 0))
L = NWindow - NOverlap
#lincoeff = []
for i in range(nframes):
	frame = w1[0+L*i:L*i+NWindow]*window
	c = NWindow-1
	corr = np.correlate(frame, frame, 'full')[c:c+order]
	R = toeplitz([corr[0:order-1]])		
	r = corr[1:order]
	r = r[:, np.newaxis]
	Rinv = np.linalg.pinv(R)
	a = -Rinv.dot(r)
	#lincoeff = np.append(lincoeff, a, axis = 0)
	lincoeff = np.column_stack((lincoeff, a))
	#energy of the error signal 
	e = corr[0]+np.sum(a*r)
	A = np.array([[1]])
	A = np.append(A, a)
#	A = np.array([[1], [a]])
	w, Af = signal.freqz(1, A, 201)#half the window points	
	ps = np.square(abs(Af))
	psl = 10*np.log10(ps)
	psl = psl[:, np.newaxis]
	psLog = np.column_stack((psLog, psl))	

#psLog = psLog.T
t = range(nframes)
plt.pcolormesh(t, w, psLog)
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [rad/s]')
plt.show()
