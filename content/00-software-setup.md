# Setting Up Programming Environment



This page provides instructions for installing the required packages and their dependencies on a local computer or server.



## Install miniforge


If you already have a preferred way to manage Python versions and libraries, you can stick to that. Otherwise, we recommend you to install Python3 and all required libraries using [Miniforge](https://conda-forge.org/download/), a free minimal installer for the package, dependency, and environment manager [Conda](https://docs.conda.io/en/latest/index.html).
- Please follow the [installation instructions](https://conda-forge.org/download/) to install Miniforge.
- After installation, verify that Conda is installed correctly by running:
```console
$ conda --version

# Example output: conda 24.11.2
```



## Configure programming environment


With Conda installed, run the command below to install all required packages and depenencies
```console
$ conda env create --yes -f https://raw.githubusercontent.com/ENCCS/practical-machine-learning/main/content/env/environment.yml
```

This will create a new environment called ``practical_machine_learning``, which can be activated with 
```console
$ conda activate practical_machine_learning
```

:::{warning}

Remember to activate your programming environment each time before running code examples. This ensures that the correct Python version and all required dependencies are available. If you forget to activate it, you may encounter errors or missing packages.
:::



## Validate programming environment


Once the programming environment is fully set up, open a new terminal (just as you should do each time before running code examples), activate the programming environment, and launch JupyterLab by running the command below.
```console
$ conda activate practical_machine_learning

$ jupyter lab
```

This will start a Jupyter server and automatically open the JupyterLab interface in your web browser.

To verify that all required packages are properly installed, follow these steps:
- Open JupyterLab (see instructions above).
- Create a new Jupyter Notebook by selecting **File** → **New** → **Notebook**.
- Copy the code examples listed below into a cell of the notebook.
- Run the cell (press **Shift + Enter** or click the **Run** button).

```python
import numpy;      print('Numpy version: ',            numpy.__version__)
import pandas;     print('Pandas version: ',           pandas.__version__)
import scipy;      print('Scipy version: ',            scipy.__version__)
import matplotlib; print('Matplotlib version: ',       matplotlib.__version__)
import seaborn;    print('Seaborn version: ',          seaborn.__version__)
import sklearn;    print('Scikit-learn version: ',     sklearn.__version__)
import keras;      print('Keras version: ',            keras.__version__)
import tensorflow; print('Tensorflow version: ',       tensorflow.__version__)
import torch;      print('Pytorch version: ',          torch.__version__)
import umap;       print('Umap-learn version: ',       umap.__version__)
import notebook;   print('Jupyter Notebook version: ', notebook.__version__)
```

You should see output similar to the figure below. The exact package versions may vary depending on when you installed them.

:::{figure} ./env/0-verification-programming-environment.png
:align: center
:width: 80%
:::

If the code runs without errors, it means the packages are correctly installed and your programming environment is ready for use. If you encounter an error (*e.g.*, ``ModuleNotFoundError``), it indicates that a package may not have been installed properly. In that case, please double-check your programming environment setup or bring the issue to the on-boarding session for assistance.

::::{warning}
If you encounter an error like the one shown in the figure below, it usually means PyTorch is trying to load a DLL (such as fbgemm.dll), but one of its dependencies is missing or incompatible. The most common causes are a missing Microsoft Visual C++ runtime or a mismatch between the installed PyTorch build and your Python/OS environment.
:::{figure} ./env/pytorch_error.png
:align: center
:width: 80%
:::

To verify, you can open another Jupyter notebook and test PyTorch again. You should see output similar to the example below.

:::{figure} ./env/pytorch_another_test.png
:align: center
:width: 80%
:::
::::


:::{note}

If you are using VS Code, you can select the installed ``practical_machine_learning`` programming environment as follows:
- Open your project folder in VS Code.
- In the upper-right corner of the editor window (when working with a Python file or Jupyter Notebook), click on **Select Kernel**.
- From the list of **Python Environments**, locate and choose the **practical_machine_learning** environment (which you have installed earlier).
- Once selected, VS Code will use this environment for running Python code and Jupyter Notebooks, ensuring that all required packages are available.
:::

