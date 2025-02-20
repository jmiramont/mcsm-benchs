from mcsm_benchs.Benchmark import Benchmark
import numpy as np
import pytest

# @pytest.fixture
def a_method(noisy_signal):
    # Dummy method for testing QRF function of the benchmark.
    results = noisy_signal # Simply return the same noisy signals.
    return results


# Test QRF computation of the benchmark.
def test_benchmark_qrf():
    # Create a dictionary of the methods to test.
    my_methods = {
        'Method 1': a_method, 
    }
    SNRin= [0, 5, 10, 20, 30, 50]
    print(SNRin)
    benchmark = Benchmark(task = 'denoising',
                        methods = my_methods,
                        N = 256, 
                        SNRin = SNRin[::-1], 
                        repetitions = 30,
                        signal_ids=['LinearChirp', 'CosChirp',],
                        verbosity=0)
                        
    benchmark.run_test()
    results_df = benchmark.get_results_as_df() # Formats the results on a DataFrame
    results_df = results_df.iloc[:,-1:-(len(SNRin)+1):-1].to_numpy()
    snr_est = np.mean(results_df,axis=0)
    snr_error = abs(np.array(SNRin)-snr_est)
    assert np.all(snr_error<0.1), 'The noise addition is not calibrated.'

# Test QRF computation of the benchmark.
def test_benchmark_saving_and_loading():
    # Create a dictionary of the methods to test.
    my_methods = {
        'Method 1': a_method, 
    }
    SNRin= [0, 5, 10, 20, 30, 50]
    print(SNRin)
    benchmark = Benchmark(task = 'denoising',
                        methods = my_methods,
                        N = 256, 
                        SNRin = SNRin[::-1], 
                        repetitions = 30,
                        signal_ids=['LinearChirp', 'CosChirp',],
                        verbosity=0)
                        
    benchmark.run_test()
    output = benchmark.save_to_file('temp_bench')
    assert output, 'The benchmark was not saved.'

    loaded_benchmark = Benchmark.load_benchmark('temp_bench')
    assert loaded_benchmark.N == benchmark.N, 'Loaded benchmark is deficient.'


# Test QRF computation of the benchmark.
def test_benchmark_sum():
    # Create a dictionary of the methods to test.
    method_1 = {
        'Method 1': a_method, 
    }
    method_2 = {
        'Method 2': a_method, 
    }

    SNRin= [0, 5, 10, 20]
    print(SNRin)
    benchmark_1 = Benchmark(task = 'detection',
                        methods = method_1,
                        N = 256, Nsub=128, 
                        SNRin = SNRin, 
                        repetitions = 30,
                        signal_ids=['LinearChirp',],
                        verbosity=0)
    benchmark_1.run_test()

    benchmark_2 = Benchmark(task = 'detection',
                        methods = method_2,
                        N = 256, Nsub=128, 
                        SNRin = SNRin, 
                        repetitions = 30,
                        signal_ids=['LinearChirp',],
                        verbosity=0)
    benchmark_2.run_test()

    benchmark = benchmark_1+benchmark_2
    print(benchmark.methods)

    # # output = benchmark.save_to_file('temp_bench')
    # assert , 'The benchmark was not saved.'

    # loaded_benchmark = Benchmark.load_benchmark('temp_bench')
    # assert loaded_benchmark.N == benchmark.N, 'Loaded benchmark is deficient.'


# test_benchmark_qrf()