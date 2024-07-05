from mcsm_benchs.Benchmark import Benchmark
from mcsm_benchs.SignalBank import SignalBank, Signal
import numpy as np
import pytest

def test_signal_class():
    sb = SignalBank(N=1024)
    signal1 = sb.signal_linear_chirp()
    s1_mean = np.mean(signal1)
    s1_std = np.std(signal1)
    s1_var = np.var(signal1)

