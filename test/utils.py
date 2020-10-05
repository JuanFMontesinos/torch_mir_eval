from contextlib import contextmanager
from timeit import default_timer

import torch

__all__ = ['cuda_timing', 'elapsed_timer']


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def cuda_timing(func):
    def inner(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        return output, time / 1000

    return inner
