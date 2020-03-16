# Use Conda's Virtual Environment

It is **strongly suggested** to create and use [conda's virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for development of the `pypop-lso` library, as presented below:

```bash
$ conda create --prefix virtual_env
$ conda activate .\virtual_env
$ conda install --prefix .\virtual_env\ python=3.7.6
$ python --version # make sure Python is installed successfully
Python 3.7.6
$ pip install numpy==1.18.1
$ python
Python 3.7.6 (default, Jan  8 2020, 20:23:39)
>>> import numpy as np # make sure NumPy is installed successfully
>>> np.__version__
'1.18.1'
>>> exit()
$ conda deactivate
```

