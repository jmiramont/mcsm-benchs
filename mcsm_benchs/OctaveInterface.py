import importlib.util 
import numpy as np
import numbers
from oct2py import octave
import os

# Check matlab.engine is installed
# try:
#     matlab_is_present = importlib.util.find_spec('matlab')
#     if matlab_is_present:
#         import matlab.engine

# except RuntimeError:
#     print("Matlab engine or Matlab installation not found.")

class OctaveInterface():
    """ This class offers an interface between python and Matlab to seamlessly run methods in a Benchmark.
    """
    def __init__(self, octave_function_name, add2path=[], matlab_warnings=False):
        """ Creates a new MatlabInterface method that calls a Matlab function.

        Args:
            matlab_function_name (str): The Matlab function name.
            add2path (list, optional): Add new paths where to look for the function indicated. Defaults to [].
            matlab_warnings (bool, optional): When True, prints out Matlab warnings. Defaults to False.

        Returns:
            MatlabInterface: An object able to call a function implemented in Matlab.
        """
        self.octave_function_name = octave_function_name
        self.eng = octave

        # try:
        #    self.eng = matlab.engine.start_matlab()
        # except NameError:
        #     print("Matlab engine or Matlab installation not found.")
        #     return None
        
        # if not matlab_warnings:
        #     self.eng.eval("warning('off','all');", nargout=0)

        octave.addpath(os.path.join('..','src','methods'))
        octave.addpath(os.path.join('src','methods'))

        for path in add2path:
            octave.addpath(path)

    def octave_function(self, signal, *params):
        """ A wrapper of a Matlab function that receives a signal to process and a variable number of positional arguments.

        Args:
            signal (numpy.ndarray): A numpy array with a signal. 

        Returns:
            An equivalent array with the outputs of the Matlab function.
        """
        all_params = list((signal.copy(),*params))
        # params = self.pre_parameters(*all_params)
        fun_handler = getattr(self.eng, self.octave_function_name)
        outputs = fun_handler(*all_params)
        if isinstance(outputs,numbers.Number):
            return outputs
        
        if len(outputs)==1:
            outputs = outputs[0]
        else:
            outputs = [output for output in outputs]
        return np.array(outputs)
        
    # def pre_parameters(self, *params):
    #     """ Cast python types to matlab types before calling the function.

    #     Returns:
    #         list: A list of matlab types.
    #     """
    #     params_matlab = list()
    #     for param in params:
    #         if isinstance(param,np.ndarray):
    #             params_matlab.append(matlab.double(vector=param.tolist()))
    #         if isinstance(param,list) or isinstance(param,tuple):
    #             params_matlab.append(matlab.double(vector=list(param)))
    #         if isinstance(param,float):
    #             params_matlab.append(matlab.double(param))
    #         if isinstance(param,int):
    #             params_matlab.append(matlab.double(float(param)))
                
    #     return params_matlab    