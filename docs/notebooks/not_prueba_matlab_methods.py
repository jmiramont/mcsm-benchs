import matlab.engine
import numpy as np
from numpy import pi as pi
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from benchmark_tools.utilstf import *
from benchmark_tools.SignalBank import SignalBank
from benchmark_tools.benchmark_utils import MatlabInterface
from methods.method_block_tresholding import NewMethod
# import sys
# sys.path.append("src\methods")

np.random.seed(0)
# signal parameters
# signal parameters
SNRin = 30
N = 2**9
Nsub=N//2
sbank = SignalBank(N=N, Nsub= Nsub)
# s = sbank.signal_cos_chirp()
s = sbank.signal_mc_multi_linear()
# s = sbank.signal_mc_cos_plus_tone()
# s = sbank.signal_mc_modulated_tones()
# s = sbank.signal_mc_synthetic_mixture()
# s = sbank.signal_mc_synthetic_mixture_2()
# s = sbank.signal_mc_impulses()
signal, noise = add_snr(s,SNRin)
signal = s + noise*np.sqrt(N/Nsub) 

signal.tofile('output.csv', sep=',', format='%f')

# Start Matlab Engine
# eng = matlab.engine.start_matlab()
# eng.eval("addpath('src/methods')")
# signal2 = matlab.double(vector=signal.tolist())

# mlint = MatlabInterface('BlockThresholding')

# funa = mlint.matlab_function
# ret = funa(signal2, matlab.double(20.0), matlab.double(N), matlab.double(10**(-SNRin/20)))
methodml = NewMethod()
funa = methodml.method
b = funa(signal, 20.0, N, 10**(-SNRin/20))
# ret = eng.BlockThresholding(signal2, matlab.double(20.0), matlab.double(N), matlab.double(10**(-SNRin/20)))



# print(ret)
# plt.plot(ret)
# plt.show()

# b = np.array(ret[0].toarray())
# print(b)

S, _, _, _ = get_spectrogram(signal)
Srec, _, _, _ = get_spectrogram(b)


f, ax = plt.subplots(1,2)
ax[0].imshow((S))
ax[1].imshow((Srec))
plt.show()
