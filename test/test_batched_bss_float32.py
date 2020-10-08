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
        self.src = np.random.rand(B, N, 44000).astype(np.float32)
        self.est = np.random.rand(B, N, 44000).astype(np.float32)

    # CPU TESTS
    def test_bss_eval_sources_permutation_false_cpu(self):
        src = torch.from_numpy(self.src.copy())
        est = torch.from_numpy(self.est.copy())

        bss_eval_sources = torch_mir_eval.batch_separation.bss_eval_sources
        sdr_b, sir_b, sar_b, _ = bss_eval_sources(src, est, compute_permutation=False)
        sdr, sir, sar, _ = torch_mir_eval.separation.bss_eval_sources(src[0], est[0], False)
        self.assertTrue(torch.allclose(sdr, sdr_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(sir, sir_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(sar, sar_b[0], rtol=1e-4))

    def test_bss_eval_sources_permutation_true_cpu(self):
        src = torch.from_numpy(self.src.copy())
        est = torch.from_numpy(self.est.copy())

        bss_eval_sources = torch_mir_eval.batch_separation.bss_eval_sources
        sdr_b, sir_b, sar_b, perm_b = bss_eval_sources(src, est, compute_permutation=True)
        sdr, sir, sar, perm = torch_mir_eval.separation.bss_eval_sources(src[0], est[0], True)
        self.assertTrue(torch.allclose(sdr, sdr_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(sir, sir_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(sar, sar_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(perm.long(), perm_b[0]))

    # GPU TESTS
    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available')
    def test_bss_eval_sources_permutation_false_cuda(self):
        src = torch.from_numpy(self.src.copy()).cuda()
        est = torch.from_numpy(self.est.copy()).cuda()

        bss_eval_sources = torch_mir_eval.batch_separation.bss_eval_sources
        sdr_b, sir_b, sar_b, _ = bss_eval_sources(src, est, compute_permutation=False)
        sdr, sir, sar, _ = torch_mir_eval.separation.bss_eval_sources(src[0], est[0], False)
        self.assertTrue(torch.allclose(sdr, sdr_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(sir, sir_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(sar, sar_b[0], rtol=1e-4))


    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available')
    def test_bss_eval_sources_permutation_true_cuda(self):
        src = torch.from_numpy(self.src.copy()).cuda()
        est = torch.from_numpy(self.est.copy()).cuda()

        bss_eval_sources = torch_mir_eval.batch_separation.bss_eval_sources
        sdr_b, sir_b, sar_b, perm_b = bss_eval_sources(src, est, compute_permutation=True)
        sdr, sir, sar, perm = torch_mir_eval.separation.bss_eval_sources(src[0], est[0], True)
        self.assertTrue(torch.allclose(sdr, sdr_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(sir, sir_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(sar, sar_b[0], rtol=1e-4))
        self.assertTrue(torch.allclose(perm.long(), perm_b[0]))

    @unittest.skipIf(BACKPROP is False, 'System non backpropagable')
    def test_bss_eval_gradient_flow(self):
        with torch.autograd.detect_anomaly():
            src = torch.from_numpy(self.src.copy())
            est = torch.from_numpy(self.est.copy()).requires_grad_()

            bss_eval_sources = torch_mir_eval.batch_separation.bss_eval_sources
            sdr_b, sir_b, sar_b, perm_b = bss_eval_sources(src, est, compute_permutation=False)
            scalar = sdr_b.mean()
            scalar.backward()
            self.assertTrue(est.grad is not None)
