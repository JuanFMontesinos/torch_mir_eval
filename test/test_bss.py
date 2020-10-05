import unittest
from contextlib import contextmanager
from timeit import default_timer

import torch
import mir_eval.separation
import librosa
import torch_mir_eval
import numpy as np


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


class TestBSS(unittest.TestCase):
    def setUp(self) -> None:
        filename = librosa.util.example_audio_file()

        y, sr = librosa.load(filename, sr=None, duration=2.)
        # self.src = y[:44000].astype(np.float64)
        # self.src = np.stack([self.src, y[20000:64000].astype(np.float64)])
        # self.est = y[40000:84000].astype(np.float64)
        # self.est = np.stack([self.est, y[10000:54000].astype(np.float64)])

        N = 5
        self.src = np.random.rand(N, 44000).astype(np.float64)
        self.est = np.random.rand(N, 44000).astype(np.float64)

    def test_bss_eval_sources_permutation_false(self):
        src = torch.from_numpy(self.src.copy()).cuda()
        est = torch.from_numpy(self.est.copy()).cuda()
        with elapsed_timer() as elapsed:
            w = mir_eval.separation.bss_eval_sources(self.src, self.est, compute_permutation=False)
        mir_eval_timing = elapsed()
        bss_eval_sources = cuda_timing(torch_mir_eval.bss_eval_sources)
        torch_metrics, torch_timing = bss_eval_sources(src, est, compute_permutation=False)
        torch_metrics = [x.cpu().numpy() for x in torch_metrics]
        self.assertTrue(np.allclose(w, torch_metrics, rtol=1e-3))
        print(f'bss_eval_sources test...\t'
              f'Compute permutation: False\t'
              f'CPU: {mir_eval_timing:.3f}\t'
              f'GPU: {torch_timing:.3f}')

    def test_bss_eval_sources_permutation_true(self):
        src = torch.from_numpy(self.src.copy()).cuda()
        est = torch.from_numpy(self.est.copy()).cuda()
        with elapsed_timer() as elapsed:
            w = mir_eval.separation.bss_eval_sources(self.src, self.est, compute_permutation=True)
        mir_eval_timing = elapsed()
        bss_eval_sources = cuda_timing(torch_mir_eval.bss_eval_sources)
        torch_metrics, torch_timing = bss_eval_sources(src, est, compute_permutation=True)
        torch_metrics = [x.cpu().numpy() for x in torch_metrics]
        self.assertTrue(np.allclose(w, torch_metrics, rtol=1e-3))
        print(f'bss_eval_sources test...\t'
              f'compute_permutation: True\t'
              f'CPU: {mir_eval_timing:.3f}\t'
              f'GPU: {torch_timing:.3f}')
