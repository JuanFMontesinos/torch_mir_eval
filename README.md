# torch_mir_eval


[![PyPI Status](https://badge.fury.io/py/torch_mir_eval.svg)](https://badge.fury.io/py/torch_mir_eval)
[![Build Status](https://github.com/JuanFMontesinos/torch_mir_eval/workflows/CI/badge.svg)](https://github.com/JuanFMontesinos/torch_mir_eval)
[![codecov][codecov-badge]][codecov]
[![Python Versions](https://img.shields.io/pypi/pyversions/asteroid.svg)](https://pypi.org/project/asteroid/)


Pytorch implementation of [mir_eval](https://craffel.github.io/mir_eval/).
Not backpropagable.
Algorithm is ~ 5 times faster (tested using Quadro P6000 for 1s 44khz samples using torch.float64)


