import importlib.util 
import numpy as np
import numbers

try:
    matlab_is_present = importlib.util.find_spec('matlab')
    if matlab_is_present:
        import matlab.engine

except RuntimeError:
    print("Matlab engine or Matlab installation not found.")


class MatlabInterface():
    """_summary_
    """
    def __init__(self, matlab_function_name, add2path=[], matlab_warnings=False):
        """_summary_

        Args:
            matlab_function_name (_type_): _description_
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
            self.eng.eval("addpath('"+path+"')")
            self.eng.eval("addpath(genpath('"+path+"'))")
        # sys.path.insert(0, os.path.abspath('../src/methods/'))

    def matlab_function(self, signal, *params):
        """_summary_

        Args:
            signal (_type_): _description_

        Returns:
            _type_: _description_
        """
        all_params = list((signal.copy(),*params))
        params = self.pre_parameters(*all_params)
        fun_handler = getattr(self.eng, self.matlab_function_name)
        outputs = fun_handler(*params)
        if isinstance(outputs,numbers.Number):
            return outputs
        
        if len(outputs)==1:
            outputs = outputs[0].toarray()
        else:
            outputs = [output.toarray() for output in outputs]
        return np.array(outputs)
        
    def pre_parameters(self, *params):
        """_summary_

        Returns:
            _type_: _description_
        """
        params_matlab = list()
        for param in params:
            if isinstance(param,np.ndarray):
                params_matlab.append(matlab.double(vector=param.tolist()))
            if isinstance(param,list) or isinstance(param,tuple):
                params_matlab.append(matlab.double(vector=list(param)))
            if isinstance(param,float):
                params_matlab.append(matlab.double(param))
            if isinstance(param,int):
                params_matlab.append(matlab.double(float(param)))
                
        return params_matlab    