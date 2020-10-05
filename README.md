# torch_mir_eval


[![PyPI Status](https://badge.fury.io/py/torch_mir_eval.svg)](https://badge.fury.io/py/torch_mir_eval)
[![Build Status](https://github.com/JuanFMontesinos/torch_mir_eval/workflows/CI/badge.svg)](https://github.com/JuanFMontesinos/torch_mir_eval)
[![Code Coverage](https://codecov.io/gh/JuanFMontesinos/torch_mir_eval/branch/main/graph/badge.svg)](https://codecov.io/gh/JuanFMontesinos/torch_mir_eval)
[![Python Versions](https://img.shields.io/pypi/pyversions/asteroid.svg)](https://pypi.org/project/asteroid/)


Pytorch implementation of [mir_eval](https://craffel.github.io/mir_eval/).
Not backpropagable.
Algorithm is ~ 5 times faster  
Tests carried out using torch.Float64, Nvidia Quadro P6000, ADM Threadripper 1920X in a single run

**Note: `float64` is preferred to achieve high tolerance during the computation (~1e-15). Using `float32` implies a tolerance ~1e-5 and result may diverge from original's `mir_eval`**  




```
bss_eval_sources test...	Compute permutation: False	CPU: 4.324	torch-CPU: 5.121
.bss_eval_sources test...	Compute permutation: False	CPU: 9.711	GPU: 2.250
..bss_eval_sources test...	compute_permutation: True	CPU: 38.775	torch-CPU: 37.519
bss_eval_sources test...	compute_permutation: True	CPU: 54.358	GPU: 7.257
*Sources vary across tests  
** Tolerance <= 1e-3
```


## How to contribute  
- Implementing https://github.com/JuanFMontesinos/torch_mir_eval/blob/52bf8f221d1f603520a13ad792bf4d22b558452a/torch_mir_eval/separation.py#L39 toeplitz matrix in pytorch.
- Implementing any other function from original `mir_eval`
