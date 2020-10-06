import unittest

import torch
from torch.testing import assert_allclose
from scipy.linalg import toeplitz as np_toeplitz
from torch_mir_eval.toeplitz import toeplitz, batch_toeplitz


class TestToeplitz(unittest.TestCase):
    def test_toeplitz_nor(self):
        c = [1, 2, 3, 5, 4]
        np_toep = np_toeplitz(c)
        torch_toep = toeplitz(c)
        assert_allclose(torch.from_numpy(np_toep), torch_toep)

    def test_toeplitz(self):
        c, r = [1, 2, 3], [4, 3, 1, 2]
        np_toep = np_toeplitz(c, r=r)
        torch_toep = toeplitz(c, r=r)
        assert_allclose(torch.from_numpy(np_toep), torch_toep)

    def test_batch_toeplitz_nor(self):
        bc = torch.randn(3, 8)
        indirect_toep = torch.stack(
            [toeplitz(c) for c in bc],
            dim=0
        )
        direct_toep = batch_toeplitz(bc)
        assert_allclose(indirect_toep, direct_toep)

    def test_batch_toeplitz(self):
        bc = torch.randn(3, 8)
        br = torch.randn(3, 4)
        indirect_toep = torch.stack(
            [toeplitz(c, r) for c, r in zip(bc, br)],
            dim=0
        )
        direct_toep = batch_toeplitz(bc, br)
        assert_allclose(indirect_toep, direct_toep)

