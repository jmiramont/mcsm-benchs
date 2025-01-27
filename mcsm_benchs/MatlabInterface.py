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

import importlib.util
import numpy as np
import numbers

# Check matlab.engine is installed
try:
    matlab_is_present = importlib.util.find_spec("matlab")
    if matlab_is_present:
        import matlab.engine

except RuntimeError:
    print("Matlab engine or Matlab installation not found.")


class MatlabInterface:
    """This class offers an interface between python and Matlab to seamlessly run methods in a Benchmark."""

    def __init__(self, matlab_function_name, add2path=[], matlab_warnings=False):
        """Creates a new MatlabInterface method that calls a Matlab function.

        Args:
            matlab_function_name (str): The Matlab function name.
            add2path (list, optional): Add new paths where to look for the function indicated. Defaults to [].
            matlab_warnings (bool, optional): When True, prints out Matlab warnings. Defaults to False.

        Returns:
            MatlabInterface: An object able to call a function implemented in Matlab.
        """
        self.matlab_function_name = matlab_function_name

        try:
            self.eng = matlab.engine.start_matlab()
        except NameError:
            print("Matlab engine or Matlab installation not found.")
            return None

        if not matlab_warnings:
            self.eng.eval("warning('off','all');", nargout=0)

        self.eng.eval("addpath('../src/methods')")
        self.eng.eval("addpath('src/methods')")

        for path in add2path:
            self.eng.eval("addpath('" + path + "')")
            self.eng.eval("addpath(genpath('" + path + "'))")
        # sys.path.insert(0, os.path.abspath('../src/methods/'))

    def matlab_function(self, signal, *params):
        """A wrapper of a Matlab function that receives a signal to process and a variable number of positional arguments.

        Args:
            signal (numpy.ndarray): A numpy array with a signal.

        Returns:
            An equivalent array with the outputs of the Matlab function.
        """
        all_params = list((signal.copy(), *params))
        params = self.pre_parameters(*all_params)
        fun_handler = getattr(self.eng, self.matlab_function_name)
        outputs = fun_handler(*params)
        if isinstance(outputs, numbers.Number):
            return outputs

        if len(outputs) == 1:
            outputs = outputs[0].toarray()
        else:
            outputs = [output.toarray() for output in outputs]
        return np.array(outputs)

    def pre_parameters(self, *params):
        """Cast python types to matlab types before calling the function.

        Returns:
            list: A list of matlab types.
        """
        params_matlab = list()
        for param in params:
            if isinstance(param, np.ndarray):
                params_matlab.append(matlab.double(vector=param.tolist()))
            if isinstance(param, list) or isinstance(param, tuple):
                params_matlab.append(matlab.double(vector=list(param)))
            if isinstance(param, float):
                params_matlab.append(matlab.double(param))
            if isinstance(param, int):
                params_matlab.append(matlab.double(float(param)))

        return params_matlab
