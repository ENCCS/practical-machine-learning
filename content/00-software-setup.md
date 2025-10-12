# Setting Up Programming Environment



## Using Personal Computer


This section provides instructions for installing the required packages and their dependencies on a local computer or server.


### Install miniforge


If you already have a preferred way to manage Python versions and libraries, you can stick to that. Otherwise, we recommend you to install Python3 and all required libraries using [Miniforge](https://conda-forge.org/download/), a free minimal installer for the package, dependency, and environment manager [Conda](https://docs.conda.io/en/latest/index.html).
- Please follow the [installation instructions](https://conda-forge.org/download/) to install Miniforge.
- After installation, verify that Conda is installed correctly by running:
```console
$ conda --version

# Example output: conda 24.11.2
```


### Configure programming environment


With Conda installed, open the **Anaconda Prompt terminal**, and run the command below to install required packages and depenencies (except PyTorch):
```console
$ conda env create -y --file=https://raw.githubusercontent.com/ENCCS/practical-machine-learning/main/content/env/environment.yml
```


This creates a new environment called ``practical_machine_learning``.
We activate it and then install PyTorch library:
```console
$ conda activate practical_machine_learning

$ conda install -y pytorch torchvision torchaudio torchtext cpuonly -c pytorch
```


:::{warning}

Remember to activate your programming environment each time before running code examples. This ensures that the correct Python version and all required dependencies are available. If you forget to activate it, you may encounter errors or missing packages.
:::


### Validate programming environment


Once the programming environment is fully set up, open a new **Anaconda Prompt terminal** (just as you should do each time before running code examples), activate the programming environment, and launch JupyterLab by running the command below:
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


If the code runs without errors, it means that all packages are correctly installed and your programming environment is ready to use.


::::{warning}
For Windows OS users, you might encounter an error 
(``ImportError: DLL load failed while importing _C: The specified procedure could not be found``) as described below.

:::{figure} ./env/pytorch_error.png
:align: center
:width: 80%
:::

It is a very common Windows-specific PyTorch issue, and it means that the underlying C++/CUDA DLLs that torch depends on could not be loaded correctly.

You should reinstall the correct matching build via the command below.

```console
$ conda install -y pytorch torchvision torchaudio torchtext cpuonly -c pytorch
```
::::


:::{note}

If you are using VS Code, you can select the installed ``practical_machine_learning`` programming environment as follows:
- Open your project folder in VS Code.
- In the upper-right corner of the editor window (when working with a Python file or Jupyter Notebook), click on **Select Kernel**.
- From the list of **Python Environments**, locate and choose the ``practical_machine_learning`` environment (which you have installed earlier).
- Once selected, VS Code will use this environment for running Python code and Jupyter Notebooks, ensuring that all required packages are available.
:::



### (Optional) Setting Up PyTorch with GPU Support


**For Windows OS users**, if your computer has a GPU card, you can install PyTorch with GPU support.
Below are step-by-step instructions to update the ``practical_machine_learning`` programming environment.


First check your CUDA version.
Open a terminal (Linux/macOS) or PowerShell (Windows) and run:
```console
$ nvcc --version
```
If ``nvcc`` is not in your PATH, you can instead run ``nvidia-smi``.
```console
$ nvidia-smi
```


Here is the output from my Windows machine:
:::{figure} ./env/test-cuda-compiler-driver.png
:align: center
:width: 80%
:::


Second, remove any CPU-only versions of PyTorch that may have been installed (for example, those coming from Conda’s defaults or conda-forge channels), and hten install an older, CUDA-compatible version of PyTorch directly using ``pip``.
Here ``cu121`` indicates the CUDA version (12.1) that the PyTorch build was compiled with.


```console
$ conda activate practical_machine_learning

$ conda remove pytorch torchvision torchaudio torchtext

$ pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```


Third, verify your installation in a Jupyter Notebook.
Run the following command and ensure it returns ``True`` for ``torch.cuda.is_available()``.


```python
import torch

print(torch.__version__)               # 2.4.0+cu121
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce GT 1030
```


## Using Google Colab


You can also run all the code examples in tutorials using [Google Colab](https://colab.research.google.com/), a free cloud-based platform that provides Jupyter Notebook environments with preinstalled ML libraries.


### Download Jupyter Notebooks


- You can open each Jupyter Notebook (usually with the ``.ipynb`` extension) from [HERE](https://github.com/ENCCS/practical-machine-learning/tree/main/content/jupyter-notebooks), and then select **Download raw file** to save it locally.
- Alternatively, you can download the entire repository at [HERE](https://github.com/ENCCS/practical-machine-learning/tree/main) by clicking the green ``<> Code`` button and choosing **Download ZIP** file. After unzipping the downloaded ZIP file, you will find all Jupyter Notebooks in the directory **practical-machine-learning-main/content/jupyter-notebooks**.


### Upload Jupyter Notebooks to Google Drive


Sign in to your [Google Drive](https://workspace.google.com/intl/en-US/products/drive/), then upload the downloaded Jupyter Notebooks to a convenient folder.
You can simply drag and drop the files directly into Google Drive or use the option **New → File upload**.


### Open Jupyter Notebooks in Google Colab


Once uploaded, right-click the Jupyter Notebooks file in Google Drive and select **Open with → Google Colaboratory**.
This will launch the notebook in Google Colab, where you can view, edit, and run the code cells interactively.


### Connect to a Hosted Runtime


In Google Colab, go to the top-right corner and click **Connect** to link your notebook to a Google-hosted runtime environment.
If you need GPU or TPU acceleration, select **Runtime → Change runtime type**, then choose the desired hardware accelerator.


### Run the Code


After connecting, follow the instructions inside the Jupyter Notebooks.
You can run each cell individually by pressing **Shift** + **Enter**, or execute the entire notebook using **Runtime → Run all**.
