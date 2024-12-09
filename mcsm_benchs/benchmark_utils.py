""" mcms-benchs: A toolbox that provides a common framework for benchmarks of multi-component signal processing methods.

    Copyright (C) 2024  Juan Manuel Miramont-Taurel

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

from abc import ABC, abstractmethod

class MethodTemplate(ABC):
    """ An abstract class for new methods to add in a benchmark.
    """

    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def method(self):
        ...

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, id):
        self._id = id
    
    @property
    def task(self):
        return self._task
    
    @task.setter
    def task(self, task):
        self._task = task

    def get_parameters(self):
        return (((),{}),) 

# class MatlabInterface():
#     """_summary_
#     """
#     def __init__(self, matlab_function_name, add2path=[], matlab_warnings=False):
#         """_summary_

#         Args:
#             matlab_function_name (_type_): _description_
#         """
#         self.matlab_function_name = matlab_function_name
#         self.eng = matlab.engine.start_matlab()
        
#         if not matlab_warnings:
#             self.eng.eval("warning('off','all');", nargout=0)

#         self.eng.eval("addpath('../src/methods')")
#         self.eng.eval("addpath('src/methods')")
        
#         for path in add2path:
#             self.eng.eval("addpath('"+path+"')")
#             self.eng.eval("addpath(genpath('"+path+"'))")
#         # sys.path.insert(0, os.path.abspath('../src/methods/'))

#     def matlab_function(self, signal, *params):
#         """_summary_

#         Args:
#             signal (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         all_params = list((signal.copy(),*params))
#         params = self.pre_parameters(*all_params)
#         fun_handler = getattr(self.eng, self.matlab_function_name)
#         outputs = fun_handler(*params)
#         if isinstance(outputs,numbers.Number):
#             return outputs
        
#         if len(outputs)==1:
#             outputs = outputs[0].toarray()
#         else:
#             outputs = [output.toarray() for output in outputs]
#         return np.array(outputs)
        
#     def pre_parameters(self, *params):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         params_matlab = list()
#         for param in params:
#             if isinstance(param,np.ndarray):
#                 params_matlab.append(matlab.double(vector=param.tolist()))
#             if isinstance(param,list) or isinstance(param,tuple):
#                 params_matlab.append(matlab.double(vector=list(param)))
#             if isinstance(param,float):
#                 params_matlab.append(matlab.double(param))
#             if isinstance(param,int):
#                 params_matlab.append(matlab.double(float(param)))
                
#         return params_matlab    