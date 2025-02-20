from mcsm_benchs.Benchmark import Benchmark
from mcsm_benchs.SignalBank import SignalBank, Signal
import numpy as np
import pytest

# Check if the custom Signal class works.
def test_signal_class():
    sb = SignalBank(N=1024, return_signal=True)
    signal1 = sb.signal_linear_chirp()
    signal1_np = signal1.view(np.ndarray)
    
    assert np.mean(signal1) == np.mean(signal1_np), "Signal custom class not conform."
    assert np.std(signal1) == np.std(signal1_np), "Signal custom class not conform."


