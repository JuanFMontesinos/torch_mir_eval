# torch_mir_eval


[![PyPI Status](https://badge.fury.io/py/torch-mir-eval.svg)](https://badge.fury.io/py/torch-mir-eval)
[![Build Status](https://github.com/JuanFMontesinos/torch_mir_eval/workflows/CI/badge.svg)](https://github.com/JuanFMontesinos/torch_mir_eval)
[![Code Coverage](https://codecov.io/gh/JuanFMontesinos/torch_mir_eval/branch/main/graph/badge.svg)](https://codecov.io/gh/JuanFMontesinos/torch_mir_eval)
[![Python Versions](https://img.shields.io/pypi/pyversions/asteroid.svg)](https://pypi.org/project/asteroid/)


Pytorch implementation of [mir_eval](https://craffel.github.io/mir_eval/).  
Nvidia RTX 3090, ADM Threadripper 1920X for a single run




```
.bss_eval_sources test>	permutation: False	float32 	CPU: 3.004	torch-CPU: 2.512
.bss_eval_sources test>	Compute permutation: False	float32 	CPU: 2.590	GPU: 1.616
.bss_eval_sources test>	compute_permutation: True	float32 	CPU: 13.195	torch-CPU: 12.326
.bss_eval_sources test>	compute_permutation: True	float32 	CPU: 13.407	GPU: 7.779
.bss_eval_sources test>	Compute permutation: False	float64 	CPU: 2.615	torch-CPU: 4.612
.bss_eval_sources test>	Compute permutation: False	float64 	CPU: 2.578	GPU: 1.941
.bss_eval_sources test>	Compute permutation: False	float64 	CPU: 12.949	torch-CPU: 22.422
.bss_eval_sources test>	Compute permutation: False	float64 	CPU: 13.142	GPU: 9.404
*Sources vary across tests  
```
## Usage
PIP install:  
`pip install torch_mir_eval`  
----------------
The easy way (work as drop-in replacement or batches):
```
from torch_mir_eval import bss_eval_sources
N=4
S=44000
src = torch.rand(N,S).cuda()
est = torch.rand(N,S).cuda()
sdr,sir,sar,perm = bss_eval_sources(src,est,compute_permutation=True)
```
```
from torch_mir_eval import bss_eval_sources
B=3
N=4
S=44000
src = torch.rand(B,N,S).cuda()
est = torch.rand(B,N,S).cuda()
sdr,sir,sar,perm = bss_eval_sources(src,est,compute_permutation=True)
```
Just pass tensors instead of numpy arrays. Everything else is the same.  
For the batched version we follow pytorch convention of batch first.
Therefore the expected format is `b, nsrc, samples`

## How to contribute  
- Implementing any other function from the original `mir_eval`
- Addresing https://github.com/JuanFMontesinos/torch_mir_eval/blob/377fe51a6b08d43af63d6b7f7805578515713a92/torch_mir_eval/batch_separation.py#L329-L333
## Changelog  
- Version 0.1: `bss_eval_sources` function implemented  
- Version 0.2: `bss_eval_sources` is now backpropagable. There algorithm now accepts batches. 
- Version 0.3: Incorporates new PyTorch's fft package for versions>1.7 and deprecates `torch.rfft` and  
                `torch.ifft` following pytorch's roadmap.  

## Current available functions  
* Separation: 
  - `mir_eval.separation.bss_eval_sources`
  - `mir_eval.batch_separation.bss_eval_sources`
