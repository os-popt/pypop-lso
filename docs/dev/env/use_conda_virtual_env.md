# Use Conda's Virtual Environment

It is **strongly suggested** to create and use [conda's virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for developing the `pypop-lso` library, as presented below:

## On Windows 10

```bash
$ conda create --prefix virtual_env
$ conda activate .\virtual_env
$ conda install --prefix .\virtual_env\ python=3.7.6
$ python --version # make sure Python is installed successfully
Python 3.7.6
$ pip install numpy==1.18.1 setuptools==46.1.1 matplotlib==3.2.1
$ python
Python 3.7.6 (default, Jan  8 2020, 20:23:39)
>>> import numpy as np # make sure NumPy is installed successfully
>>> np.__version__
'1.18.1'
>>> import setuptools
>>> setuptools.__version__
'46.1.1'
>>> import matplotlib
>>> matplotlib.__version__
'3.2.1'
>>> exit()
$ conda deactivate
```

## On Linux

```bash
$ conda create --prefix virtual_env
$ conda activate ./virtual_env
$ conda install --prefix ./virtual_env/ python=3.7.6
$ pip install numpy==1.18.1 setuptools==46.1.1 matplotlib==3.2.1
$ conda deactivate
```
