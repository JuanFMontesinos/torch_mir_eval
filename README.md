# torch_mir_eval


[![PyPI Status](https://badge.fury.io/py/torch_mir_eval.svg)](https://badge.fury.io/py/torch_mir_eval)
[![Build Status](https://github.com/JuanFMontesinos/torch_mir_eval/workflows/CI/badge.svg)](https://github.com/JuanFMontesinos/torch_mir_eval)
[![Code Coverage](https://codecov.io/gh/JuanFMontesinos/torch_mir_eval/branch/main/graph/badge.svg)](https://codecov.io/gh/JuanFMontesinos/torch_mir_eval)
[![Python Versions](https://img.shields.io/pypi/pyversions/asteroid.svg)](https://pypi.org/project/asteroid/)


Pytorch implementation of [mir_eval](https://craffel.github.io/mir_eval/).
Not backpropagable.
Algorithm is ~ 5 times faster  
Tests carried out using torch.Float64, Nvidia Quadro P6000, ADM Threadripper 1920X in a single run
Sources vary across tests 

```
bss_eval_sources test...	Compute permutation: False	CPU: 4.324	torch-CPU: 5.121
.bss_eval_sources test...	Compute permutation: False	CPU: 9.711	GPU: 2.250
..bss_eval_sources test...	compute_permutation: True	CPU: 38.775	torch-CPU: 37.519
bss_eval_sources test...	compute_permutation: True	CPU: 54.358	GPU: 7.257
```


