import unittest

import torch
import mir_eval.separation
import torch_mir_eval
import numpy as np

from utils import *

BACKPROP = True


class TestBSS(unittest.TestCase):
    def setUp(self) -> None:
        N = 5
        self.src = np.random.rand(N, 44000).astype(np.float64)
        self.est = np.random.rand(N, 44000).astype(np.float64)

    # CPU TESTS
    def test_bss_eval_sources_permutation_false_cpu(self):
        src = torch.from_numpy(self.src.copy())
        est = torch.from_numpy(self.est.copy())
        with elapsed_timer() as elapsed:
            w = mir_eval.separation.bss_eval_sources(self.src, self.est, compute_permutation=False)
        mir_eval_timing = elapsed()
        bss_eval_sources = torch_mir_eval.separation.bss_eval_sources
        with elapsed_timer() as elapsed:
            torch_metrics = bss_eval_sources(src, est, compute_permutation=False)
        torch_timing = elapsed()
        torch_metrics = [x.cpu().numpy() for x in torch_metrics]
        self.assertTrue(np.allclose(w, torch_metrics))
        print(f'bss_eval_sources test>\t'
              f'Compute permutation: False\t'
              f'float64 \t'
              f'CPU: {mir_eval_timing:.3f}\t'
              f'torch-CPU: {torch_timing:.3f}')

    def test_bss_eval_sources_permutation_true_cpu(self):
        src = torch.from_numpy(self.src.copy())
        est = torch.from_numpy(self.est.copy())
        with elapsed_timer() as elapsed:
            w = mir_eval.separation.bss_eval_sources(self.src, self.est, compute_permutation=True)
        mir_eval_timing = elapsed()
        bss_eval_sources = torch_mir_eval.separation.bss_eval_sources
        with elapsed_timer() as elapsed:
            torch_metrics = bss_eval_sources(src, est, compute_permutation=True)
        torch_timing = elapsed()
        torch_metrics = [x.cpu().numpy() for x in torch_metrics]
        self.assertTrue(np.allclose(w, torch_metrics))
        print(f'bss_eval_sources test>\t'
              f'Compute permutation: False\t'
              f'float64 \t'
              f'CPU: {mir_eval_timing:.3f}\t'
              f'torch-CPU: {torch_timing:.3f}')

    # GPU TESTS
    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available')
    def test_bss_eval_sources_permutation_false_cuda(self):
        src = torch.from_numpy(self.src.copy()).cuda()
        est = torch.from_numpy(self.est.copy()).cuda()
        with elapsed_timer() as elapsed:
            w = mir_eval.separation.bss_eval_sources(self.src, self.est, compute_permutation=False)
        mir_eval_timing = elapsed()
        bss_eval_sources = cuda_timing(torch_mir_eval.separation.bss_eval_sources)
        torch_metrics, torch_timing = bss_eval_sources(src, est, compute_permutation=False)
        torch_metrics = [x.cpu().numpy() for x in torch_metrics]
        self.assertTrue(np.allclose(w, torch_metrics))
        print(f'bss_eval_sources test>\t'
              f'Compute permutation: False\t'
              f'float64 \t'
              f'CPU: {mir_eval_timing:.3f}\t'
              f'GPU: {torch_timing:.3f}')

    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available')
    def test_bss_eval_sources_permutation_true_cuda(self):
        src = torch.from_numpy(self.src.copy()).cuda()
        est = torch.from_numpy(self.est.copy()).cuda()
        with elapsed_timer() as elapsed:
            w = mir_eval.separation.bss_eval_sources(self.src, self.est, compute_permutation=True)
        mir_eval_timing = elapsed()
        bss_eval_sources = cuda_timing(torch_mir_eval.separation.bss_eval_sources)
        torch_metrics, torch_timing = bss_eval_sources(src, est, compute_permutation=True)
        torch_metrics = [x.cpu().numpy() for x in torch_metrics]
        self.assertTrue(np.allclose(w, torch_metrics, rtol=1e-3))
        print(f'bss_eval_sources test>\t'
              f'Compute permutation: False\t'
              f'float64 \t'
              f'CPU: {mir_eval_timing:.3f}\t'
              f'GPU: {torch_timing:.3f}')
