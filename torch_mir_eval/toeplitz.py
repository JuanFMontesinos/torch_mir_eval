import torch


def toeplitz(c, r=None):
    """
    Construct a Toeplitz matrix.

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == c`` is
    assumed.
    Inspired from scipy.linalg.toeplitz


    Parameters
    ----------
    c : tensor_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : tensor_like, optional
        First row of the matrix. If None, ``r = c`` is assumed.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.

    Returns
    -------
    A : (len(c), len(r)) torch.Tensor
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

    Examples
    --------
    >>> toeplitz([1,2,3], [1,4,5,6])
    tensor([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])

    """
    c = torch.as_tensor(c).view(-1)
    r = c if r is None else torch.as_tensor(r).view(-1)
    vals = torch.cat([torch.flip(c, dims=(0,)), r[1:]])
    toep = torch.stack([
        torch.roll(vals, shifts=i, dims=0)[-len(r):] for i in range(len(c))
    ])
    return toep


def batch_toeplitz(c, r=None):
    """
    Construct a batch Toeplitz matrix.

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == c`` is
    assumed.
    Inspired from scipy.linalg.toeplitz

    Parameters
    ----------
    c : tensor_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a batched 1-D array.
    r : tensor_like, optional
        First row of the matrix. If None, ``r = c`` is assumed.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[:, 0], r[:, 1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a batched 1-D array.

    Returns
    -------
    A : (batch, len(c), len(r)) torch.Tensor
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

    Examples
    --------
    >>> import torch
    >>> batch_toeplitz(torch.randn(2, 3), torch.randn(2, 4))
    """
    c = torch.as_tensor(c).view(c.shape[0], -1)
    r = c if r is None else torch.as_tensor(r).view(r.shape[0], -1)
    assert c.shape[0] == r.shape[0], "Batch dimension shoudl agree between r and c."

    vals = torch.cat([torch.flip(c, dims=(1,)), r[:, 1:]], dim=1)
    toep = torch.stack([
        torch.roll(vals, shifts=i, dims=1)[:, -r.shape[1]:] for i in range(c.shape[1])
    ], dim=1)
    return toep
