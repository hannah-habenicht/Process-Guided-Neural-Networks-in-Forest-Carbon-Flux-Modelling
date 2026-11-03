
# PRELES Extension

This repository provides a C++ extension for integrating the **PRELES** process-based model into PyTorch workflows. The PRELES model simulates forest ecosystem processes such as **Gross Primary Production (GPP)**, **Evapotranspiration (ET)**, and **Soil Water (SW)** dynamics. A detailed description of PRELES can be found in [Peltoniemi et al. (2015)](https://jukuri.luke.fi/handle/10024/485901). The code is provided in the [here](https://github.com/MikkoPeltoniemi/Rpreles/tree/master/src).

This extension enables efficient, differentiable access to the PRELES model for hybrid machine learning applications, including calibration and physics-informed training.

It is based on the [work](https://github.com/biometry/PiNNs/blob/main/src/README.md) of Marieke Wesselkamp and Niklas Moser.


## Build Instructions
This extension is written in C++ and compiled using PyTorch's CppExtension mechanism. To build the extension:
navigate to the folder

`python setup.py build_ext --inplace`
then run 
"pip install ."

### Prerequisites
Depends on the remaining 
Python 3.8+

PyTorch >= 1.8

C++17 compatible compiler

setuptools

### File Structure
`setup.py` – Build script using setuptools and torch.utils.cpp_extension

`call_preles.cpp` – Python binding for the main PRELES entry point

`initruns.cpp` – Initialization routines for PRELES state variables

`preles.cpp` – Main logic implementing the PRELES process model

`gpp.cpp` – Gross Primary Production (GPP) computation

`water.cpp` – Hydrological balance and evapotranspiration components

### Compilation Flags
The extension is compiled with debugging support enabled:

`extra_compile_args=['-g']`

This allows line-by-line debugging of the C++ model during development. You can remove -g for production builds.

### Usage
The module is used in conjunction with PyTorch-based data loaders and optimization routines. Once compiled, you can call the model in Python as follows:

`import preles`

`output = preles.preles(*input_tensor)`

Where input_tensor contains the full sequence of PRELES inputs, including meteorological data and model parameters. Outputs include GPP, ET, and other internal variables as differentiable PyTorch tensors.



### License
This extension is provided for research purposes only. Please cite relevant papers if used in scientific work.