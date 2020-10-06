import unittest

import torch
import mir_eval.separation
import torch_mir_eval
import numpy as np

from utils import *

BACKPROP = True


class TestBatchedBSS(unittest.TestCase):
    def setUp(self) -> None:
        B = 3
        N = 5
        self.src = np.random.rand(B, N, 44000).astype(np.float64)
        self.est = np.random.rand(B, N, 44000).astype(np.float64)

    # GPU TESTS
    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available')
    def test_bss_eval_sources_permutation_false_cuda(self):
        src = torch.from_numpy(self.src.copy()).cuda()
        est = torch.from_numpy(self.est.copy()).cuda()

        bss_eval_sources = cuda_timing(torch_mir_eval.batch_separation.bss_eval_sources)
        torch_metrics, torch_timing = bss_eval_sources(src, est, compute_permutation=False)
        sdr, sir, sar, _ = torch_mir_eval.separation.bss_eval_sources(src[0], est[1], False)
        print('')
