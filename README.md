# torch_mir_eval


[![PyPI Status](https://badge.fury.io/py/torch-mir-eval.svg)](https://badge.fury.io/py/torch-mir-eval)
[![Build Status](https://github.com/JuanFMontesinos/torch_mir_eval/workflows/CI/badge.svg)](https://github.com/JuanFMontesinos/torch_mir_eval)
[![Code Coverage](https://codecov.io/gh/JuanFMontesinos/torch_mir_eval/branch/main/graph/badge.svg)](https://codecov.io/gh/JuanFMontesinos/torch_mir_eval)
[![Python Versions](https://img.shields.io/pypi/pyversions/asteroid.svg)](https://pypi.org/project/asteroid/)


Pytorch implementation of [mir_eval](https://craffel.github.io/mir_eval/).
Algorithm is ~ 5 times faster  
Nvidia Quadro P6000, ADM Threadripper 1920X for a single run




```
bss_eval_sources test...	Compute permutation: False	CPU: 3.834	torch-CPU: 4.696
bss_eval_sources test...	Compute permutation: False	CPU: 3.902	GPU: 1.629
bss_eval_sources test...	compute_permutation: True	CPU: 21.478	torch-CPU: 22.758
bss_eval_sources test...	compute_permutation: True	CPU: 19.259	GPU: 7.834
*Sources vary across tests  
```
## Usage
`pip install torch_mir_eval`  
```
import torch_mir_eval as mir_eval
#enjoy
```
Just pass tensors instead of numpy arrays. Everything else is the same.  

## How to contribute  
- Implementing any other function from the original `mir_eval`

## Changelog  
- Version 0.1: `bss_eval_sources` function implemented  

## Current available functions  
* Separation: 
  - `mir_eval.separation.bss_eval_sources`:
