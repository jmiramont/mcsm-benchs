"""mcms-benchs: A toolbox that provides a common framework for benchmarks of multi-component signal processing methods.

Copyright (C) 2024  Juan Manuel Miramont

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from mcsm_benchs.SignalBank import SignalBank, Signal
import pandas as pd
import numbers
import pickle

# from functools import partial
import multiprocessing

# from parallelbar import progress_map
import time

from numpy import mean, abs
from numpy.linalg import norm

from tqdm import tqdm


class Benchmark:
    """
    This class performs a number of tasks for methods comparison. It abstracts a benchmark itself.

    """

    def __init__(
        self,
        task="denoising",
        methods=None,
        N=256,
        Nsub=None,
        parameters=None,
        SNRin=None,
        repetitions=None,
        signal_ids="all",
        verbosity=1,
        parallelize=False,
        complex_noise=False,
        obj_fun=None,
        write_log=True,
        name=None,
        description=None,
        **kwargs,
    ):
        """Initialize the main parameters of the test bench before running the benchmark.

        Args:
            task (str, optional): The task to test the methods. Defaults to 'denoising'.

            methods (dict, optional): A dictionary of functions. Defaults to None.

            N (int, optional): Lengths of the observation window for the signal
            generation. Defaults to 256.

            Nsub (int, optional): Lengths of the signal. If None, uses Nsub (See
            SignalBank class description for details). Defaults to None.

            parameters (dict, optional): A dictionary of parameters for the methods
            to run. The keys of this dictionary must be same as the methods dictionary.
            Defaults to None.

            SNRin (tuple, optional): List or tuple with the SNR values.
            Defaults to None.

            repetitions (int, optional): Number of times each method is applied for
            each value of SNR. This value is the number of noise realizations that are
            used to assess the methods. Defaults to None.

            signal_ids (tuple, optional): Tuple or list of the signal ids from the
            SignalBank class. Defaults to 'all'.

            verbosity (int, optional): Number from 0 to 4. It determines the number of
            messages passed to the console informing the progress of the benchmarking
            process. Defaults to 1.

            parallelize (bool, optional): If True, tries to run the process in parallel.
            Defaults to False.

            complex_noise (bool, optional): If True, uses complex noise. A function can
            be passed to compute a new realization of noise instead.
            Defaults to False.

            obj_fun (callable, optional): If None, used the default objective functions
            for benchmarking. If a function is passed as an argument, the default is
            overridden.

            write_log (bool, optional): If True, saves a log of errors and warnings. Defaults to True.

            name (str, optional): A name to identify the benchmark in collaborative repositories.

            description (str, optional): A description to show in collaborative repositories.

        """

        self.write_log = write_log
        self.log = []

        self.name = name
        self.description = description

        # Check input parameters and initialize the object attributes
        self.input_parsing(
            task,
            methods,
            N,
            Nsub,
            parameters,
            SNRin,
            repetitions,
            signal_ids,
            verbosity,
            parallelize,
            complex_noise,
            obj_fun,
            **kwargs,
        )

        # Parallelize parameters
        if self.parallel_flag:
            if self.verbosity > 1:
                print("Number of processors: ", multiprocessing.cpu_count())
                print("Parallel pool: {}".format(self.processes))

    def __add__(self, other):
        """Overload the addition operator for Benchmark objects with the same parameters.

        Args:
            other (Benchmark): A Benchmark with the same parameters and possibly different methods.

        Raises:
            TypeError:

        Returns:
            Benchmark: A Benchmark object with the combined methods.
        """
        if isinstance(other, Benchmark):
            return self.sum(self, other)
        else:
            raise TypeError(
                "Unsupported operand type(s) for +: '{}' and '{}'".format(
                    type(self), type(other)
                )
            )

    def input_parsing(
        self,
        task,
        methods,
        N,
        Nsub,
        parameters,
        SNRin,
        repetitions,
        signal_ids,
        verbosity,
        parallelize,
        complex_noise,
        obj_fun,
        **kwargs,
    ):
        """Parse input parameters of the constructor of class Benchmark.

        Args:
            task (str, optional): The task to test the methods. Defaults to 'denoising'.
            methods (dict, optional): A dictionary of functions. Defaults to None.
            N (int, optional): Lengths of the signals. Defaults to 256.
            parameters (dict, optional): A dictionary of parameters for the methods
            to run. The keys of this dictionary must be same as the methods dictionary.
            Defaults to None.
            SNRin (tuple, optional): List or tuple with the SNR values.
            Defaults to None.
            repetitions (int, optional): Number of times each method is applied for
            each value of SNR.
            This value is the number of noise realizations that are used to assess the
            methods.Defaults to None.
            signal_ids (tuple, optional): Tuple or list of the signal ids from the
            SignalBank class. Defaults to 'all'.
            verbosity (int, optional): Number from 0 to 4. It determines the number of
            messages passed to the console informing the progress of the benchmarking
            process. Defaults to 1.
            parallelize (bool, optional): If True, tries to run the process in parallel.
            Defaults to False.

        Raises:
            ValueError: If any parameter is not correctly parsed.
        """
        # Check verbosity
        assert (
            isinstance(verbosity, int) and 0 <= verbosity < 6
        ), "Verbosity should be an integer between 0 and 5"
        self.verbosity = verbosity

        # TODO: ADD NEW TASKS
        # Check the task is either 'denoising' or 'detection'.
        # if (task != 'denoising' and task != 'detection'):
        #     raise ValueError("The tasks should be either 'denoising' or 'detection'.\n")
        # else:
        self.task = task

        # Check methods is a dictionary
        if type(methods) is not dict:
            raise ValueError("Methods should be a dictionary.\n")
        else:
            self.methods = methods
            self.methods_ids = [llave for llave in methods]

        # If no parameters are given to the benchmark.
        if parameters is None:
            self.parameters = {key: (((), {}),) for key in methods.keys()}
        else:
            if type(parameters) is dict:
                self.parameters = parameters
            else:
                raise ValueError("Parameters should be a dictionary or None.\n")

        # Check both dictionaries have the same keys:
        if not (self.methods.keys() == self.parameters.keys()):
            # sys.stderr.write
            raise ValueError(
                "Both methods and parameters dictionaries should have the same keys.\n"
            )

        # Check if N is an entire:
        if type(N) is int:
            self.N = N
        else:
            raise ValueError("N should be an entire.\n")

        # Check if Nsub is an entire:
        if Nsub is not None:
            if type(Nsub) is not int:
                raise ValueError("Nsub should be an entire.\n")

            # Check if Nsub is lower than N:
            if self.N < Nsub:
                raise ValueError("Nsub should be lower than N.\n")

        self.Nsub = Nsub

        # Check if SNRin is a tuple or list, and if so, check if there are only numerical variables.
        if (type(SNRin) is tuple) or (type(SNRin) is list):
            for i in SNRin:
                if not isinstance(i, numbers.Number):
                    raise ValueError("All elements in SNRin should be real numbers.\n")

            self.SNRin = SNRin
        else:
            raise ValueError("SNRin should be a tuple or a list.\n")

        # Check if repetitions is an entire:
        if type(repetitions) is int:
            self.repetitions = repetitions
        else:
            raise ValueError("Repetitions should be an entire.\n")

        # Check what to do with the signals:

        # Check if signal_ids is a dict with testing signals
        if isinstance(signal_ids, dict):
            signal_dic = {}
            for key in signal_ids:
                s = signal_ids[key]
                assert N == len(s), "Input signal length should be N"
                if isinstance(signal_ids[key],Signal):
                    signal_dic[key] = s
                else:
                    signal_dic[key] = Signal(s)
                

            self.signal_dic = signal_dic
            self.signal_ids = signal_ids
            self.tmin = 0  # int(np.sqrt(N))
            self.tmax = N  # -int(np.sqrt(N))

        else:
            if signal_ids == "all":
                # Generates a dictionary of signals
                signal_bank = SignalBank(N=self.N, Nsub=self.Nsub, return_signal=True)
                self.tmin = signal_bank.tmin  # Save initial and end times of signals.
                self.tmax = signal_bank.tmax

                self.signal_dic = signal_bank.signalDict
                self.signal_ids = [akey for akey in self.signal_dic]
                if Nsub is None:
                    self.Nsub = signal_bank.Nsub

            else:
                # Check if list of signals are in SignalBank
                if isinstance(signal_ids, tuple) or isinstance(signal_ids, list):
                    signal_bank = SignalBank(
                        N=self.N, Nsub=self.Nsub, return_signal=True
                    )
                    llaves = signal_bank.signalDict.keys()
                    assert all(signal_id in llaves for signal_id in signal_ids)

                self.signal_dic = signal_bank.signalDict
                self.tmin = signal_bank.tmin  # Save initial and end times of signals.
                self.tmax = signal_bank.tmax
                self.signal_ids = signal_ids

        # Check if complex_noise flag is bool:
        if type(complex_noise) is bool:
            self.complex_noise = complex_noise
        else:
            if callable(complex_noise):
                self.complex_noise = complex_noise
            else:
                if complex_noise != "NF":
                    raise ValueError("'complex_noise' should be a bool or callable.\n")

        # Handle parallelization parameters:
        max_processes = multiprocessing.cpu_count()

        if parallelize is False:
            self.parallel_flag = False
        else:
            if parallelize is True:
                if parallelize:
                    available_proc = multiprocessing.cpu_count()
                    self.processes = np.max((1, available_proc // 2))
                    self.parallel_flag = True
            else:
                if isinstance(parallelize, int):
                    if parallelize < max_processes:
                        self.processes = parallelize
                    else:
                        self.processes = max_processes
                    self.parallel_flag = True

        # Check objective function
        # Set the default performance function according to the selected task
        if obj_fun is None:
            self.objectiveFunction = {
                "perf_metric": self.set_objective_function(task),
            }
        else:
            if callable(obj_fun):
                self.objectiveFunction = {
                    "obj_fun": obj_fun,
                }
            else:
                if type(obj_fun) is dict:
                    self.objectiveFunction = obj_fun
                else:
                    raise ValueError(
                        "'obj_fun' should be a callable object or a dictionary.\n"
                    )

        # TODO Check this list of attribute initialization
        # Extra arguments may be passed when opening a saved benchmark:
        if kwargs == {}:
            # If we are here, this is a new benchmark, so the all the methods are new:
            self.this_method_is_new = {method: True for method in self.methods_ids}
            self.results = None
            self.noise_matrix = None
            self.methods_and_params_dic = dict()

        else:
            for key in kwargs:
                if not key in self.__dict__:
                    self.__dict__[key] = kwargs[key]

    def check_methods_output(self, output, input):
        """Check that the outputs of the method to benchmark fulfill the required type
        and shape.

        Args:
            output: Output from the method. The type and shape depends on the task.
            input: Input passed to the method to produce output.

        Raises:
            ValueError: If the output does not comply with the required type and shape
            for the selected task.
        """
        if self.task == "denoising":
            if type(output) is not np.ndarray:
                raise ValueError(
                    "Method's output should be a numpy array for task='denoising'.\n"
                )

            if output.shape != input.shape:
                raise ValueError(
                    "Method's output should have the same shape as input for task='denoising'.\n"
                )

    def set_objective_function(self, task):
        """
        Set the performance function for the selected task (future tasks could easily add new performance functions)
        """

        compFuncs = {  #'denoising': lambda x: self.snr_comparison(*x,tmin=self.tmin,tmax=self.tmax),
            "denoising": self.snr_comparison,
            "detection": self.detection_perf_function,
            "component_denoising": self.compare_qrf_block,
            "inst_frequency": self.compare_instf_block,
            "misc": None,
        }
        return compFuncs[task]

    def inner_loop(self, benchmark_parameters, timer=False):
        """Main loop of the Benchmark.

        Args:
            benchmark_parameters (tuple): Tuple or list with the parameters of the benchmark.
            timer (bool): If true, measures the time of execution.

        Returns:
            narray: Return a numpy array, the shape of which depends on the selected task.
        """

        method, params, idx = benchmark_parameters

        if self.verbosity >= 5:
            print("------ Inner loop. " + method + ": " + str(idx), flush=True)

        # Get the noisy signal (as a ndarray) and wrap it with the Signal class, adding
        # the signal information from the base signal.
        # This wrapper class behaves like a numpy array, but encapsulates signal info,
        # like the total number of components or number of components per time.

        noisy_signal = Signal(self.noisy_signals[idx])
        noisy_signal.ncomps = self.base_signal_info["ncomps"]
        noisy_signal.total_comps = self.base_signal_info["total_comps"]
        noisy_signal.instf = self.base_signal_info["instf"]

        try:
            args, kwargs = params
            tstart = time.time()
            method_output = self.methods[method](noisy_signal, *args, **kwargs)
            elapsed = time.time() - tstart

        except BaseException as err:
            print(
                f"Unexpected error {err=}, {type(err)=} in method {method}. Watch out for NaN values."
            )

            # Write an entry log for errors.
            if self.write_log:
                log_entry = f"Unexpected error {err=}, {type(err)=} in method {method}. Noise matrix index: {idx=}."
                self.log.append(log_entry)

            elapsed = np.nan

            if self.task == "denoising":
                method_output = np.empty(noisy_signal.shape)
                method_output[:] = np.nan

            if self.task == "detection":
                method_output = np.nan

            if self.task == "misc":
                method_output = np.nan

        #! Rewrite this part.
        # self.check_methods_output(method_output,noisy_signals) # Just checking if the output its valid.

        return method_output, elapsed

    def set_results_dict(self):
        """Initializes a dictionary where the results are going to be saved later."""
        # If its a new benchmark...
        if self.results is None:
            self.results = dict()
            self.elapsed_time = dict()

            # Create nested dictionary for results and elapsed time:
            for fun_name in self.objectiveFunction:
                self.results[fun_name] = {}
                for signal_id in self.signal_ids:
                    self.results[fun_name][signal_id] = {}
                    self.elapsed_time[signal_id] = {}
                    for SNR in self.SNRin:
                        self.results[fun_name][signal_id][SNR] = {}
                        for method in self.methods:
                            self.results[fun_name][signal_id][SNR][method] = {}
                            self.elapsed_time[signal_id][method] = {}
                            # for param in self.parameters[method]:
                            #     self.results[signal_id][SNR][method][str(param)] = {}
        else:
            # Add new methods to the dictionary of results
            print("- Rerun benchmark.")
            print('-New methods',[method for method in self.methods if method not in self.results[self.objectiveFunction.keys()[0]][self.signal_ids.keys()[0]][self.SNRin[0]].keys()])

            for fun_name in self.objectiveFunction:
                for signal_id in self.signal_ids:
                    for SNR in self.SNRin:
                        for method in self.methods:
                            if (
                                method
                                not in self.results[fun_name][signal_id][SNR].keys()
                            ):
                                # print('-- New Method',method)
                                self.results[fun_name][signal_id][SNR][method] = {}
                                self.elapsed_time[signal_id][method] = {}

    def run_test(self): ## Make this a deprecation warning.
        print(
            "Method run_test() will be deprecated in newer versions. Use run() instead."
        )
        self.run()

    def run(self):
        """Run the benchmark.

        Returns:
            dict: Returns nested dictionaries with the results of the benchmark.
        """
        if self.verbosity >= 0:
            print("Running benchmark...")
            if self.verbosity > 1:
                bar_fun = lambda smth: smth
            else:
                bar_fun = lambda smth: tqdm(smth)

        # If it is a new benchmark, create a nested dict() to save the results.
        # If it is an old benchmark with new methods, create the new keys.
        self.set_results_dict()

        # This run all the experiments and save the results in nested dictionaries.
        for signal_id in self.signal_ids:
            if self.verbosity >= 1:
                print("- Signal " + signal_id)

            self.base_signal = self.signal_dic[signal_id]
            self.base_signal_info = self.signal_dic[signal_id].get_info()

            for SNR in bar_fun(self.SNRin):
                if self.verbosity >= 2:
                    print("-- SNR: {} dB".format(SNR))

                # If the benchmark has been run before,
                # re-run again with the same noise.
                if self.noise_matrix is None:
                    self.noise_matrix = self.generate_noise()

                noisy_signals, scaled_noise = self.sigmerge(
                    self.base_signal,
                    self.noise_matrix,
                    SNR,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    return_noise=True,
                )

                # Access current noisy signals from the main loop.
                self.noisy_signals = noisy_signals

                # ===========================MAIN LOOP===================================

                # ------------------------- Parallel loop ------------------------------
                if self.parallel_flag:
                    parallel_list = list()
                    for method in self.methods:
                        if self.this_method_is_new[method]:
                            if self.verbosity >= 2:
                                print(
                                    "--- Parallel loop -- Method: "
                                    + method
                                    + "(all parameters)"
                                )
                            for p, params in enumerate(self.parameters[method]):
                                args, kwargs = get_args_and_kwargs(params)
                                for idx, noisy_signal in enumerate(noisy_signals):
                                    parallel_list.append([method, (args, kwargs), idx])

                    # Here implement the parallel stuff
                    pool = multiprocessing.Pool(processes=self.processes)
                    parallel_results = pool.map(self.inner_loop, parallel_list)
                    pool.close()
                    pool.join()
                    if self.verbosity >= 2:
                        print("--- Parallel loop finished.")

                # ---------------------- Serial loop -----------------------------------
                k = 0  # This is used to get the parallel results if it's necessary.

                for method in self.methods:

                    if self.this_method_is_new[method]:
                        if self.verbosity >= 3:
                            print("--- Method: " + method)

                        for p, params in enumerate(self.parameters[method]):
                            elapsed = []

                            if self.verbosity >= 4:
                                print("---- Parameters Combination: " + str(p))

                            args, kwargs = get_args_and_kwargs(params)

                            if self.task == "component_denoising":
                                extrargs = {"tmin": self.tmin, "tmax": self.tmax}
                                method_output = []

                            if self.task == "inst_frequency":
                                extrargs = {"tmin": self.tmin, "tmax": self.tmax}
                                # method_output = [[] for aaa in range(noisy_signals.shape[0])]
                                method_output = []

                            if self.task == "denoising":
                                extrargs = {"tmin": self.tmin, "tmax": self.tmax}
                                # method_output = np.zeros_like(noisy_signals)
                                method_output = []
                            if self.task == "detection":
                                extrargs = {}
                                # method_output = np.zeros((self.repetitions)).astype(bool)
                                method_output = []

                            if self.task == "misc":
                                method_output = []

                            for idx, noisy_signal in enumerate(noisy_signals):
                                # Get results from parallel computation
                                if self.parallel_flag:
                                    tmp, extime = parallel_results[k]
                                    # method_output[idx] = tmp
                                    method_output.append(tmp)
                                    # Save but DON'T TRUST the exec. time in parallel.
                                    elapsed.append(extime)
                                    k += 1

                                # Or from serial computation
                                else:
                                    tmp, extime = self.inner_loop(
                                        [method, (args, kwargs), idx]
                                    )
                                    # method_output[idx] = tmp
                                    method_output.append(tmp)
                                    elapsed.append(extime)

                            # Either way, results are saved in a nested dictionary-----
                            # For each performance metric
                            for fun_name in self.objectiveFunction:
                                result = []
                                for i, output in enumerate(method_output):
                                    try:
                                        fun = self.objectiveFunction[fun_name]
                                        perf_met_output = fun(
                                            self.base_signal,
                                            output,
                                            tmin=self.tmin,
                                            tmax=self.tmax,
                                            scaled_noise=scaled_noise[i],
                                        )

                                    except BaseException as err:
                                        print(
                                            f"Unexpected error {err=}, {type(err)=} while applying performance metric:{fun_name}. Watch out for NaN values."
                                        )
                                        perf_met_output = np.nan

                                    result.append(perf_met_output)

                                # Saving results -----------
                                if np.any([type(r) == dict for r in result]):
                                    aux = {key: [] for key in result[0]}
                                    for r in result:
                                        for key in r:
                                            aux[key].append(r[key])
                                    result = aux

                                self.results[fun_name][signal_id][SNR][method][
                                    str(params)
                                ] = result
                                self.elapsed_time[signal_id][method][
                                    str(params)
                                ] = elapsed

                            if self.verbosity > 4:
                                print("Elapsed:{}".format(np.mean(elapsed)))

                        self.methods_and_params_dic[method] = [
                            str(key) for key in self.parameters[method]
                        ]

        if self.verbosity > 0:
            print("The test has finished.")

        # Don't use old methods if run again.
        for method in self.this_method_is_new:
            self.this_method_is_new[method] = False

        return self.results

    def save_to_file(self, filename=None):
        """Save the results to a binary file that encodes the benchmark object.
        Notice that the methods associated with the benchmark, not being pickable objects,
        are NOT saved.

        Args:
            filename (str, optional): Path and filename. Defaults to None.

        Returns:
            bool: True if the file was successfully created.
        """

        if filename is None:
            filename = "a_benchmark"

        a_copy = self
        a_copy.methods = {key: None for key in a_copy.methods}
        a_copy.base_signal = a_copy.base_signal.view(np.ndarray)
        # a_copy.signal_dic = []
        a_copy.noisy_signals = a_copy.noisy_signals.view(np.ndarray)
        a_copy.objectiveFunction = {key: None for key in a_copy.objectiveFunction}

        if callable(a_copy.complex_noise):
            a_copy.complex_noise = "NF"

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(a_copy.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

        return True

    def get_results_as_df(self, results=None):
        """Get a pandas DataFrame object with the results of the benchmark.

        Args:
            results (dict, optional): Nested dictionary with the results of the
            benchmark. Defaults to None.

        Returns:
            DataFrame: Returns a pandas DataFrame with the results.
        """
        if list(self.results.keys())[0] in self.signal_ids:
            self.results = {"perf_metric": self.results}

        if results is None:
            df = self.dic2df(self.results)
        else:
            df = self.dic2df(results)

        df = pd.concat({param: df[param] for param in df.columns})
        df = df.unstack(level=3)
        df = df.reset_index()
        df.columns = pd.Index(
            ["Parameter", "Perf.Metric", "Signal_id", "Method", "Repetition"]
            + self.SNRin
        )
        df = df.reindex(
            columns=["Perf.Metric", "Method", "Parameter", "Signal_id", "Repetition"]
            + self.SNRin
        )
        df = df.sort_values(by=["Method", "Parameter", "Signal_id"])

        data_frames = []
        for per_fun in np.unique(df["Perf.Metric"]):
            dfaux = df[df["Perf.Metric"] == per_fun].iloc[:, 1::]
            aux2 = np.zeros((dfaux.shape[0],), dtype=bool)
            for metodo in self.methods_and_params_dic:
                aux = np.zeros((dfaux.shape[0],), dtype=bool)
                for params in self.methods_and_params_dic[metodo]:
                    aux = aux | (
                        (dfaux["Parameter"] == params) & (dfaux["Method"] == metodo)
                    )
                aux2 = aux2 | aux

            df2 = dfaux[aux2]
            data_frames.append(df2)

        if len(data_frames) == 1:
            return data_frames[0]
        else:
            return data_frames

    def add_new_method(self, methods, parameters=None, perf_func=None):
        """Add new methods to an existing Benchmark.

        Args:
            methods (_type_): A dictionary of methods.
            parameters (_type_, optional): If necessary, a dictionary of parameters for the new methods, or new parameters to explore with already benchmarked methods. Defaults to None.
        """
        # Check methods is a dictionary and update existing dictionary of methods.
        if type(methods) is not dict:
            raise ValueError("Methods should be a dictionary.\n")

        # If no parameters are given to the benchmark.
        if parameters is None:
            parameters = {key: (((), {}),) for key in methods}
        else:
            if type(parameters) is not dict:
                raise ValueError("Parameters should be a dictionary or None.\n")

        for key in methods:
            if key not in self.methods.keys():
                self.methods[key] = methods[key]
                self.methods_ids.append(key)
                self.parameters[key] = parameters[key]
                # self.elapsed_time[key]  = dict()
                self.this_method_is_new[key] = True

        # Check both dictionaries have the same keys:
        if not (self.methods.keys() == self.parameters.keys()):
            # sys.stderr.write
            raise ValueError(
                "Both methods and parameters dictionaries should have the same keys.\n"
            )

        self.objectiveFunction = perf_func
        # New methods cannot be parallelized (for now).
        self.parallel_flag = False

    # Other functions:
    def dic2df(self, mydic):
        """Get a pandas DataFrame from a nested dictionary.

        Args:
            mydic (dict): A nested dictionary of results coming from a Benchmark object.

        Returns:
            DataFrame : A DataFrame with a column per-level in the dictionary.
        """
        auxdic = dict()
        for key in mydic:
            if isinstance(mydic[key], dict):
                df = self.dic2df(mydic[key])
                auxdic[key] = df
            else:
                return pd.DataFrame(mydic)

        df = pd.concat(auxdic, axis=0)
        return df

    def generate_noise(self):
        """Generate a matrix of noise realizations of shape [self.repetitions,self.N].
        One realization per row.

        Returns:
            numpy.array: Matrix with noise realizations.
        """

        if isinstance(self.complex_noise, bool):
            np.random.seed(0)
            noise_matrix = np.random.randn(self.repetitions, self.N)
            if self.complex_noise:
                noise_matrix = np.random.rand(
                    self.repetitions, self.N
                ) + 1j * np.random.randn(self.repetitions, self.N)

        if callable(self.complex_noise):
            noise_matrix = np.random.randn(self.repetitions, self.N)
            for i in range(self.repetitions):
                noise_matrix[i, :] = self.complex_noise(self.N)

        return noise_matrix

    # Static methods--------------------------------------------------------------------
    @staticmethod
    def load_benchmark(filename, **kwargs):
        """Load a Benchmark object from a file saved using the class method.

        Args:
            filename (str): A path to the saved benchmark.

        Returns:
            Benchmark: A Benchmark object.
        """
        with open(filename + ".pkl", "rb") as f:
            benchmark_dict = pickle.load(f)

        return Benchmark(**benchmark_dict)

    @staticmethod
    def sigmerge(x1, noise, snr, tmin=None, tmax=None, return_noise=False):
        """Merge a signal and a noise realization with a given SNR (in dB).

        Args:
            x1 (_type_): _description_
            noise (_type_): _description_
            ratio (_type_): _description_
            tmin (_type_, optional): _description_. Defaults to None.
            tmax (_type_, optional): _description_. Defaults to None.
            return_noise (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_

        Example:
            >>> import mcsm_benchs.Benchmark
            >>> import mcsm_benchs.SignalBank
            >>> sb = SignalBank(N=1024)
            >>> signal = sb.signal_linear_chirp()
            >>> noise = np.random.randn(1024,)
            >>> snr = 10 # In dB
            >>> mixture = Benchmark.sigmerge(signal,noise,snr=snr)
        """
        # Get signal parameters.
        N = len(x1)

        if tmin is None:
            tmin = 0
        if tmax is None:
            tmax = N

        sig = np.random.randn(*noise.shape)
        ex1 = np.mean(np.abs(x1[tmin:tmax]) ** 2)

        if len(noise.shape) == 1:
            ex2 = np.mean(np.abs(noise) ** 2)
        else:
            ex2 = np.mean(np.abs(noise) ** 2, axis=1)
        h = np.sqrt(ex1 / (ex2 * 10 ** (snr / 10)))

        if len(noise.shape) > 1:
            h.resize((noise.shape[0], 1))

        scaled_noise = noise * h
        # sig = sig*h.reshape((noise.shape[0],1))
        sig = x1 + scaled_noise

        if return_noise:
            return sig, scaled_noise
        else:
            return sig

    @staticmethod
    def snr_comparison(x, x_hat, tmin=None, tmax=None, **kwargs):
        """
        Quality reconstruction factor for denoising performance characterization.
        """
        if tmin is None:
            tmin = 0

        if tmax is None:
            tmax = len(x)

        x = x[tmin:tmax]

        if len(x_hat.shape) == 1:
            x_hat = x_hat[tmin:tmax]
            qrf = 10 * np.log10(np.sum(x**2) / np.sum((x_hat - x) ** 2))
        else:
            x_hat = x_hat[:, tmin:tmax]
            qrf = np.zeros((x_hat.shape[0],))
            for i in range(x_hat.shape[0]):
                qrf[i] = 10 * np.log10(np.sum(x**2) / np.sum((x_hat[i, :] - x) ** 2))

        return qrf

    @staticmethod
    def detection_perf_function(original_signal, detection_output, **kwargs):
        """Performance function for detection. Returns the boolean output of the detection methods."""
        return detection_output

    @staticmethod
    def compare_qrf_block(signal, method_output, tmin=None, tmax=None, **kwargs):
        """ Compare the output SNR, i.e. qualitiy reconstruction factor (QRF) in a vectorized way for multi-component outputs.

        Args:
            signal (numpy.array or Signal): _description_
            method_output (numpy.array): _description_
            tmin (int, optional): tmin and tmax define a window of the signal to compare SNR. Defaults to None.
            tmax (int, optional): tmin and tmax define a window of the signal to compare SNR. Defaults to None.

        Returns:
            dict : A dictionary with the QRF of each component.
        """
        X = signal.comps
        # output = []
        # for Xest in method_output:
        order = order_components(method_output, X)
        Xaux = [method_output[i] for i in order]
        qrfs = []
        for x, xaux in zip(X, Xaux):
            indx = np.where(np.abs(x) > 0)
            qrfs.append(compute_qrf(x[indx], xaux[indx], tmin=tmin, tmax=tmax))
        # output.append(qrfs)
        # output = np.array(output, dtype=object)
        dict_output = {"Comp.{}".format(i): qrfs[i] for i in range(len(qrfs))}
        return dict_output

    @staticmethod
    def compare_instf_block(signal, method_output, tmin=None, tmax=None, **kwargs):
        """ Compute the instantaneous frequency for multi-component outputs.

        Args:
            signal (numpy.array or Signal): _description_
            method_output (numpy.array): _description_
            tmin (int, optional): tmin and tmax define a window of the signal to compare SNR. Defaults to None.
            tmax (int, optional): tmin and tmax define a window of the signal to compare SNR. Defaults to None.

        Returns:
            dict: A dictionary with the IF of each component.
        """
        X = signal.instf
        # output = []
        # for Xest in method_output:
        order = order_components(method_output, X, minormax="min", metric=mse)
        Xaux = [method_output[i] for i in order]
        qrfs = []
        for x, xaux in zip(X, Xaux):
            indx = np.where(np.abs(x) > 0)
            # qrfs.append(compute_qrf(x[indx], xaux[indx],tmin=tmin,tmax=tmax))
            qrfs.append(mse(x[indx], xaux[indx]))
        # output.append(qrfs)
        # output = np.array(output, dtype=object)
        dict_output = {"Comp.{}".format(i): qrfs[i] for i in range(len(qrfs))}
        return dict_output

    @staticmethod
    def sum(bench_a, bench_b):
        """ This function is used to sum to benchmarks by overloading the + operator.
        Summing benchmark means transfer the results of bench_b to bench_a as long as they only differ on the methods/parameters used.

        Args:
            bench_a (Benchmark): The first summand.
            bench_b (Benchmark): The second summand.

        Returns:
            Benchmark: A Benchmark with the combined methods of bench_a and bench_b.
        """
        assert (
            bench_a.repetitions == bench_b.repetitions
        ), "Repetitions should be same in both benchmarks."

        assert (
            bench_a.SNRin == bench_b.SNRin
        ), "SNRin should be the same in both benchmarks."

        assert bench_a.N == bench_b.N, "N should be the same in both benchmarks."

        assert (
            bench_a.Nsub == bench_b.Nsub
        ), "Nsub should be the same in both benchmarks."

        assert np.all(
            bench_a.signal_dic.keys() == bench_b.signal_dic.keys()
        ), "Signals must be the same in both benchmarks."

        assert np.all(
            [key for key in bench_a.objectiveFunction]
            == [key for key in bench_b.objectiveFunction]
        ), "Benchmarks must use the same performance functions."

        bench_c = bench_a

        # Transfer results
        for fun_name in bench_c.objectiveFunction:
            for SNR in bench_c.SNRin:
                for signal_id in bench_c.signal_ids:
                    for method in bench_b.methods:
                        if method not in bench_c.methods.keys():
                            # bench_c.methods[method] = bench_b.methods[method]
                            # bench_c.methods_ids.append(method)
                            # bench_c.parameters[method] = bench_b.parameters[method]
                            # bench_c.this_method_is_new[method] = False
                            bench_c.results[fun_name][signal_id][SNR][method] = {}
                            bench_c.elapsed_time[signal_id][method] = {}
                            for params in bench_b.parameters[method]:
                                bench_c.results[fun_name][signal_id][SNR][method][
                                    str(params)
                                ] = bench_b.results[fun_name][signal_id][SNR][method][
                                    str(params)
                                ]
                                # bench_c.elapsed_time[signal_id][key][str(params)] = bench_b.elapsed_time[signal_id][key][str(params)]

        for method in bench_b.methods:
            if method not in bench_c.methods.keys():
                bench_c.methods[method] = bench_b.methods[method]
                bench_c.methods_ids.append(method)
                bench_c.parameters[method] = bench_b.parameters[method]
                bench_c.this_method_is_new[method] = False
                bench_c.methods_and_params_dic[method] = bench_b.methods_and_params_dic[
                    method
                ]
        return bench_c


""" 
----------------------------------------------------------------------------------------
END OF BENCHMARK CLASS DEFINITION
----------------------------------------------------------------------------------------
"""

"""
----------------------------------------------------------------------------------------
Other auxiliary functions
----------------------------------------------------------------------------------------
"""

def mse(x, xest):
    """ Mean square error performance function.

    Args:
        x (numpy.array): The original noiseless signal.
        xest (numpy.array): An estimation of x.

    Returns:
        float: Mean square error between x and xest.
    """
    assert len(x) == len(xest), "Should be of equal length."
    idx = np.where(abs(x) > 0)
    x_aux = x[idx]
    xest_aux = xest[idx]
    error = np.mean((x_aux - xest_aux) ** 2)
    return error


def corr_comps(x, xest):
    """ Normalized correlation between x and xest.

    Args:
        x (numpy.array): The original noiseless signal.
        xest (numpy.array): An estimation of x.

    Returns:
        float: Normalized correlation (between -1 and 1).
    """
    idx = np.where(abs(x) > 0)
    x_aux = x[idx]
    xest_aux = xest[idx]
    cor = abs(sum((x_aux - mean(x_aux)) * (xest_aux - mean(xest_aux)))) / (
        norm(x_aux - mean(x_aux)) * norm(xest_aux - mean(xest_aux)) + 1e-15
    )
    return cor


def order_components(Xest, X, minormax="max", metric=corr_comps):
    """ This functions receives a multicomponent output Xest of a method and find a correspondence with the original noiseless components X by minimizing (or maximizing) the metric. 

    Args:
        Xest (numpy.array): _description_
        X (numpy.array): _description_
        minormax (str, optional): 'max' or 'min' according to what is needed from the given metric. Defaults to "max".
        metric (Callable, optional): A function `m = fun(x,xest)`, with m a real number, and vectors x and xest are the noiseless component x and the estimation of x correspondingly. Defaults to correlation between components.

    Returns:
        numpy.array: A vector of length K, with the correspondence x[0]<->xest[order[0]].
    """
    order = [[] for aaa in range(len(X))]
    values = np.array([[metric(x, xest) for x in X] for xest in Xest], dtype=object)
    if minormax == "max":
        fun = np.argmax
        factor = -1
    if minormax == "min":
        fun = np.argmin
        factor = 1

    while np.any(np.array([k == [] for k in order])):
        ind = np.unravel_index(fun(values, axis=None), values.shape)
        # if (ind[0] not in order) and (order[ind[1]] == []):
        if (order[ind[1]] == []):
            order[ind[1]] = int(ind[0])
        values[ind] = factor * np.inf
    return order


def compute_qrf(x, x_hat, tmin=None, tmax=None):
    """
    Quality reconstruction factor
    """
    if tmin is None:
        tmin = 0

    if tmax is None:
        tmax = len(x)

    x = x[tmin:tmax]
    x_hat = x_hat[tmin:tmax]
    qrf = 10 * np.log10(np.sum(x**2) / np.sum((x_hat - x) ** 2))
    return qrf


def get_args_and_kwargs(params):
    if type(params) is dict:
        args = []
        kwargs = params
    else:
        dict_indicator = [type(i) is dict for i in params]
        if any(dict_indicator):
            assert (
                len(params) == 2
            ), "Parameters must be given as a dictionary or an iterable."
            for i in range(len(params)):
                kwargs = params[np.where(dict_indicator)[0][0]]
                args = params[np.where([not i for i in dict_indicator])[0][0]]
        else:
            args = params
            kwargs = dict()

    return args, kwargs
