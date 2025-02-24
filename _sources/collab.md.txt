
# Collaborative benchmarks

One main purpose of `mcsm-benchs` is to serve as the base of \emph{collaborative} benchmarks (a concept in {cite}`benchopt`), growing with the support of the community in order to be representative and useful.
One step towards this direction is the use of open-source software.

Another step towards a more general adoption of this good practice is to make the benchmarks generated using `mcsm-benchs` available to the community using public repositories of code (like GitHub, GitLab, Codeocean, etc).
These services provide tools designed to foster cooperation between interested members of the community.
They also allow other users to easily add new methods and raise issues to be improved in the benchmarks.
We thus provide a custom GitHub [template repository](https://github.com/jmiramont/collab-benchmark-template), which can be used to generate new collaborative benchmarks.
Below we show a typical directory tree of such a repository.

                            a-benchmark-repository
                            ├─ results
                            ├─ src
                            │  ├─ methods
                            │  └─ utilities
                            ├─ config.yaml
                            ├─ publish_results.py
                            └─ run_this_benchmark.py

## Configuration file
A configuration file `config.yaml` is included in the repository template, where the definition of the simulation parameters is encoded, as shown in the following example:

```yaml
# config.yaml file contents. 
# Fix benchmark parameters:
N: 1024                 # Signal length
SNRin: [-5, 0, 10, 20]  # SNRs to test
repetitions: 100        # Number of simulations
parallelize: False      # Run the benchmark in parallel

# Show all messages from the benchmark
verbosity: 5            

# Run again the same benchmark but with new methods:
add_new_methods: False
```

With `add_new_methods:False` the benchmark is run from scratch, with the purpose of reproducing a whole set of simulations.
By choosing `add_new_methods:True` when new methods are added to a preexisting benchmark, simulations run applying only the new approaches, extending previous results.

The completed benchmarks are saved in the `results` folder.
A website hosted in the repository, which is deployed using continuous integration and deployment workflows, is generated automatically using the functions provided in the class `ResultsInterpreter`.
From this site, the authors can access interactive plots that summarize the results, and download the whole set of comparisons, for each signal or the entire benchmark, in a `.csv` file format.

## Adding new methods to collaborative benchmarks

In order to add a new method, a user just needs to add a `.py` file representing the method in the `methods` folder (even for `MATLAB`/`Octave`-implemented approaches).
The file should define a class that encapsulates the method, and should inherit from a custom abstract class `MethodTemplate` designed to force the implementation of specific class methods. 
[Template files](https://github.com/jmiramont/collab-benchmark-template/tree/main/new_method_examples) and [step-by-step guides](https://github.com/jmiramont/collab-benchmark-template/tree/main/new_method_examples) are provided to make this procedure very straightforward.
An example of the content of such a file is shown below.

```python
# Make sure this file's name starts with "method_*.py"

# Import here all the modules you need and the template class
from mcsm_benchs.benchmark_utils import MethodTemplate

# Create here a new class that will encapsulate your method.
# This class should inherit the abstract class MethodTemplate.

class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'a_new_method'    # Choose a name
        self.task = 'misc'          # and a task

    # Must implement this function representing your method
    def method(self, signal, *args, **kwargs): 
        ...

    # Optionally, implement this to pass parameters to your method:
    def get_parameters(self,):
        # # Example: Combination of positional/keyword arguments:
        # return [((5, 6),{'a':True,'b':False}), # One set of parameters.
        #        ((2, 1),{'a':False,'b':True}), # Another set of parameters.    
        #        ]
```

The class function `get_parameters(...)` should return a list/tuple of input arguments for the method. 
The positional arguments must be indicated in a tuple or a list, whereas the keyword arguments must be indicated using a dictionary as is custom in `Python`.
If the method uses either just positional (resp. keyword) arguments, leave an empty tuple (resp. dictionary).

To collaborate with a public benchmark, interested authors can first add the file that represents their approach to the `src/methods` folder in a local copy --commonly a *fork*-- of an existing benchmark repository.
Then, maintainers of the public benchmark can accept the changes via a *pull-request*.
This way, public benchmarks can be updated.
Such functionality makes it easier for a team to work together, each team member providing different methods to be compared later in a benchmark.
Furthermore, the benchmark can be periodically run in the cloud, independently of the contributors, using continuous integration tools.

Once all methods are defined, the benchmark can be run using `run_this_benchmark.py`, which automatically discovers all methods in the repository and launches the benchmark with the parameters given in the configuration file `config.yaml`.

## Public online reports

The `ResultsInterpreter` class can generate reports summarizing the main benchmark parameters and results.
These can be made available online, along with interactive figures and `.csv` files, which are also produced by the `ResultsInterpreter` class to complement the report.

When a collaborative benchmark is created using the GitHub repository template, the outputs of the `ResultsInterpreter` class are automatically used by the continuous integration workflow to generate and publish an online report.
Users can interact with figures showing the results in the website or download the \code{.csv} files for further exploration.
The goal of this tool is to complement scientific articles, easing the access to much more information than what normally fits in an article, and encouraging reproducible science.
An example of the automatic website created with the results can be seen online [here](https://jmiramont.github.io/signal-detection-benchmark).
