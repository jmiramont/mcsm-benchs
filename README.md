# A Toolbox for Benchmarking of Multi-Component Signal Methods

We introduce a public, open-source, Python-based toolbox for benchmarking multi-component signal analysis methods, implemented either in Python or Matlab.

The goal of this toolbox is providing the signal-processing community with a common framework that allows researcher-independent comparisons between methods and favors reproducible research.

## Modify ```matlabengine``` module version

Check the version of the ```matlabengine``` module you have to install to use run the benchmarks in the next table:

| Matlab Version | Python Version | ```matlabengine``` version |   |
|----------------|----------------|----------------------------|---|
| 2022b          | 3.8, 3.9, 3.10 | 9.13.16                    |   |
| 2022a          | 3.8, 3.9       | 9.12.17                    |   |
| 2021b          | 3.7, 3.8, 3.9  | 9.11.19                    |   |

Then, look for the ```matlabengine``` line in [```pyproject.toml```](./pyproject.toml), it should look like this:

```python
matlabengine = "9.12.17"
```

Make sure to change the version with the one corresponding to your Python and Matlab current versions. If you have an older version of Matlab or Python, you can search for the correct version of the ```matlabengine``` module [here](https://pypi.org/project/matlabengine/#history).

After this, run

```bash
poetry lock
```

## Size of outputs according to the task

The shape and type of the output depends on the task.

- For Denoising: The output must be a vector array with the same length as the signal.
- For Mode Retrieval: The output must be an array of size ```[J,N]```, where ```J``` is the number of components, and ```N``` is the length of the signal.
- For Instantaneous Frequency: The output must be an array of size ```[J,N]```, where ```J``` is the number of components, and ```N``` is the length of the signal. Each row of the array represents the estimation of the instantaneous frequency of the signal.
