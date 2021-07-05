# -*- coding: utf-8 -*-
'''
Source separation algorithms attempt to extract recordings of individual
sources from a recording of a mixture of sources.  Evaluation methods for
source separation compare the extracted sources from reference sources and
attempt to measure the perceptual quality of the separation.

See also the bss_eval MATLAB toolbox:
    http://bass-db.gforge.inria.fr/bss_eval/

Conventions
-----------

An audio signal is expected to be in the format of a 1-dimensional array where
the entries are the samples of the audio signal.  When providing a group of
estimated or reference sources, they should be provided in a 2-dimensional
array, where the first dimension corresponds to the source number and the
second corresponds to the samples.

Metrics
-------

* :func:`torch_mir_eval.separation.bss_eval_sources`: Computes the bss_eval_sources
  metrics from bss_eval, which optionally optimally match the estimated sources
  to the reference sources and measure the distortion and artifacts present in
  the estimated sources as well as the interference between them.

References
----------
  .. [#vincent2006performance] Emmanuel Vincent, Rémi Gribonval, and Cédric
      Févotte, "Performance measurement in blind audio source separation," IEEE
      Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.


'''

import itertools
import warnings

import torch
from math import log2, ceil
from torch.fft import fft, ifft

from .toeplitz import batch_toeplitz

# The maximum allowable number of sources (prevents insane computational load)
MAX_SOURCES = 100

__all__ = ['bss_eval_sources']


def validate(reference_sources: torch.Tensor, estimated_sources: torch.Tensor):
    """Checks that the input data to a metric are valid, and throws helpful
    errors if not.

    Parameters
    ----------
    reference_sources : torch.Tensor, shape=(B,nsrc, nsampl)
        matrix containing true sources
    estimated_sources : torch.Tensor, shape=(B,nsrc, nsampl)
        matrix containing estimated sources

    """

    if reference_sources.shape != estimated_sources.shape:
        raise ValueError(f'The shape of estimated sources and the true '
                         f'sources should match.  reference_sources.shape '
                         f'= {reference_sources.shape}, estimated_sources.shape '
                         f'= {estimated_sources.shape}')
    n_dim = reference_sources.ndim
    B, nsrc, nsampl = reference_sources.shape
    if n_dim > 3:
        raise ValueError(f'The number of dimensions is too high (must be less '
                         f'than 3). reference_sources.ndim = {reference_sources.ndim}, '
                         f'estimated_sources.ndim '
                         f'= {estimated_sources.ndim}')

    if reference_sources.size == 0:
        warnings.warn("reference_sources is empty, should be of size "
                      "(nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty torch.Tensors")
    elif _any_source_silent(reference_sources):
        raise ValueError('All the reference sources should be non-silent (not '
                         'all-zeros), but at least one of the reference '
                         'sources is all 0s, which introduces ambiguity to the'
                         ' evaluation. (Otherwise we can add infinitely many '
                         'all-zero sources.)')

    if estimated_sources.size == 0:
        warnings.warn("estimated_sources is empty, should be of size "
                      "(B,nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty torch.Tensors")
    elif _any_source_silent(estimated_sources):
        raise ValueError('All the estimated sources should be non-silent (not '
                         'all-zeros), but at least one of the estimated '
                         'sources is all 0s. Since we require each reference '
                         'source to be non-silent, having a silent estimated '
                         'source will result in an underdetermined system.')

    if B > MAX_SOURCES:
        raise ValueError(f'The supplied matrices should be of shape (nsrc,'
                         f' nsampl) but reference_sources.shape[0] = {B} and '
                         f'estimated_sources.shape[0] = {B} which is greater '
                         f'than mir_eval.separation.MAX_SOURCES = {MAX_SOURCES}.  To '
                         f'override this check, set '
                         f'mir_eval.separation.MAX_SOURCES to a '
                         f'larger value.')


def _any_source_silent(sources):
    """Returns true if the parameter sources has any silent first dimensions"""
    return torch.any(torch.sum(sources, dim=-1) == 0)


def bss_eval_sources(reference_sources, estimated_sources,
                     compute_permutation=True):
    """
    Ordering and measurement of the separation quality for estimated source
    signals in terms of filtered true source, interference and artifacts.

    The decomposition allows a time-invariant filter distortion of length
    512, as described in Section III.B of [#vincent2006performance]_.

    Passing ``False`` for ``compute_permutation`` will improve the computation
    performance of the evaluation; however, it is not always appropriate and
    is not the way that the BSS_EVAL Matlab toolbox computes bss_eval_sources.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, sir, sar,
    ...  perm) = torch_mir_eval.separation.bss_eval_sources(reference_sources,
    ...                                               estimated_sources)

    Parameters
    ----------
    reference_sources : torch.Tensor, shape=(b, nsrc, nsampl)
        matrix containing true sources (must have same shape as
        estimated_sources)
    estimated_sources : torch.Tensor, shape=(b, nsrc, nsampl)
        matrix containing estimated sources (must have same shape as
        reference_sources)
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)

    Returns
    -------
    sdr : torch.Tensor, shape=(b, nsrc)
        vector of Signal to Distortion Ratios (SDR)
    sir : torch.Tensor, shape=(b, nsrc)
        vector of Source to Interference Ratios (SIR)
    sar : torch.Tensor, shape=(b, nsrc)
        vector of Sources to Artifacts Ratios (SAR)
    perm : torch.Tensor, shape=(b, nsrc)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number ``perm[j]`` corresponds to
        true source number ``j``). Note: ``perm`` will be ``[0, 1, ...,
        nsrc-1]`` if ``compute_permutation`` is ``False``.

    References
    ----------
    .. [#] Emmanuel Vincent, Shoko Araki, Fabian J. Theis, Guido Nolte, Pau
        Bofill, Hiroshi Sawada, Alexey Ozerov, B. Vikrham Gowreesunker, Dominik
        Lutter and Ngoc Q.K. Duong, "The Signal Separation Evaluation Campaign
        (2007-2010): Achievements and remaining challenges", Signal Processing,
        92, pp. 1928-1936, 2012.

    """

    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[None, None, :]
    elif estimated_sources.ndim == 2:
        estimated_sources = estimated_sources[None, :]

    if reference_sources.ndim == 1:
        reference_sources = reference_sources[None, None, :]
    elif reference_sources.ndim == 2:
        reference_sources = reference_sources[None, :]
    validate(reference_sources, estimated_sources)

    # Permuting to shape nsrc,b,nsample for readability
    estimated_sources = estimated_sources.permute(1, 0, 2).contiguous()
    reference_sources = reference_sources.permute(1, 0, 2).contiguous()
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

    nsrc, b, nsamples = estimated_sources.shape

    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = torch.empty((nsrc, nsrc, b))
        sir = torch.empty((nsrc, nsrc, b))
        sar = torch.empty((nsrc, nsrc, b))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt(reference_sources,
                                        estimated_sources[jest],
                                        jtrue, 512)
                # Compute this outside the loop is memory greedy
                # Storing (nsrc, nsrc, b) vs (nsrc, nsrc, b, nsamples)
                sdr[jest, jtrue], sir[jest, jtrue], sar[jest, jtrue] = \
                    _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # select the best ordering
        batch_idx = list(range(b))
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_sir = torch.empty(len(perms), b)
        dum = torch.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_sir[i] = torch.mean(sir[perm, dum].view(-1, b), dim=0)
        popt = torch.Tensor(perms)[torch.argmax(mean_sir, dim=0)].long()
        sdr = sdr[popt, dum][batch_idx, :, batch_idx].squeeze()
        sir = sir[popt, dum][batch_idx, :, batch_idx].squeeze()
        sar = sar[popt, dum][batch_idx, :, batch_idx].squeeze()
        return sdr, sir, sar, popt
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = torch.empty(nsrc, b)
        sir = torch.empty(nsrc, b)
        sar = torch.empty(nsrc, b)
        for j in range(nsrc):
            s_true, e_spat, e_interf, e_artif = \
                _bss_decomp_mtifilt(reference_sources,
                                    estimated_sources[j],
                                    j, 512)
            sdr[j], sir[j], sar[j] = \
                _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # return the default permutation for compatibility
        popt = torch.arange(nsrc).unsqueeze(1).expand_as(sdr)
        return sdr.T.squeeze(), sir.T.squeeze(), sar.T.squeeze(), popt.T.squeeze()


def _bss_decomp_mtifilt(reference_sources: torch.Tensor, estimated_source: torch.Tensor, j: int, flen: int):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    """
    nsrc, b, nsampl = reference_sources.shape
    # decomposition
    # true source image
    s_true = torch.cat((reference_sources[j], torch.zeros(b, flen - 1,
                                                          device=reference_sources.device,
                                                          dtype=reference_sources.dtype)), dim=-1)
    # spatial (or filtering) distortion
    e_spat = _project(reference_sources[j, None, ...], estimated_source,
                      flen) - s_true
    # interference
    e_interf = _project(reference_sources,
                        estimated_source, flen) - s_true - e_spat  # artifacts
    e_artif = -s_true - e_spat - e_interf
    e_artif[:, :nsampl] += estimated_source
    return s_true, e_spat, e_interf, e_artif


def _fix_shape(x, n, axis):
    """ Internal auxiliary function for _raw_fft, _raw_fftnd."""
    s = list(x.shape)

    index = [slice(None)] * len(s)
    index[axis] = slice(0, s[axis])
    s[axis] = n
    z = torch.zeros(s, dtype=x.dtype, device=x.device)
    z[tuple(index)] = x
    return z


def _calc_G(sf, b, nsrc, flen, **kw):
    G = torch.zeros((b, nsrc * flen, nsrc * flen), **kw)
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = sf[i] * torch.conj(sf[j])
            ssf = ifft(ssf).real
            ss = batch_toeplitz(
                c=torch.cat((ssf[:, 0].unsqueeze(1), ssf.flip(1)[:, 0:flen - 1]), dim=-1),
                r=ssf[:, :flen].unsqueeze(1))
            G[:, i * flen: (i + 1) * flen, j * flen: (j + 1) * flen] = ss
            G[:, j * flen: (j + 1) * flen, i * flen: (i + 1) * flen] = ss.permute(0, 2, 1)

    return G


def _calc_D(sf, sef, b, nsrc, flen, **kw):
    D = torch.zeros(b, nsrc * flen, **kw)
    for i in range(nsrc):
        ssef = sf[i] * torch.conj(sef)
        ssef = ifft(ssef).real
        D[:, i * flen: (i + 1) * flen] = torch.cat((ssef[:, 0].unsqueeze(1), ssef.flip(1)[:, 0:flen - 1]), dim=-1)
    return D


def _project(reference_sources, estimated_source, flen):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """

    nsrc, b, nsampl = reference_sources.shape
    def_kw = {'device': reference_sources.device, 'dtype': torch.float64}
    kw = {'device': reference_sources.device, 'dtype': reference_sources.dtype}

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = torch.cat((reference_sources,
                                   torch.zeros((nsrc, b, flen - 1), **kw)), dim=-1)

    estimated_source = torch.cat((estimated_source, torch.zeros(b, flen - 1, **kw)), dim=-1)
    n_fft = int(2 ** ceil(log2(nsampl + flen - 1.)))
    rs = _fix_shape(reference_sources, n_fft, -1)  # Padding like scipy.fftpack.fft does
    sf = fft(rs)
    es = _fix_shape(estimated_source, n_fft, -1)  # Padding like scipy.fftpack.fft does
    sef = fft(es)
    # inner products between delayed versions of reference_sources
    G = _calc_G(sf, b, nsrc, flen, **kw)
    # inner products between estimated_source and delayed versions of
    # reference_sources

    D = _calc_D(sf, sef, b, nsrc, flen, **kw)
    # Computing projection
    # Distortion filters
    # TODO case in which only few of the Gs are singular but the rest are determined
    if all(torch.det(G) > 0.1):
        C = torch.linalg.solve(G, D.unsqueeze(-1)).reshape(b, nsrc, flen).permute(0, 2, 1)
    else:
        solutions = []
        for d, g in zip(D, G):
            c = torch.linalg.lstsq(g, d.unsqueeze(1)).solution.reshape(nsrc, flen).T
            solutions.append(c)
        C = torch.stack(solutions)
    # Filtering
    sproj = torch.zeros(b, nsampl + flen - 1, **kw)
    for j in range(b):
        for i in range(nsrc):
            sproj[j, ...] += torch.nn.functional.conv1d(reference_sources[i, j][:-flen + 1][None, None, ...],
                                                        C[j, :, i].flip([0])[None, None, ...],
                                                        padding=flen - 1)[0, 0]

    return sproj


def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = 10 * torch.log10(torch.sum(s_filt ** 2, dim=-1) / torch.sum((e_interf + e_artif) ** 2, dim=-1))
    sir = 10 * torch.log10(torch.sum(s_filt ** 2, dim=-1) / torch.sum(e_interf ** 2, dim=-1))
    sar = 10 * torch.log10(torch.sum((s_filt + e_interf) ** 2, dim=-1) / torch.sum(e_artif ** 2, dim=-1))
    return sdr, sir, sar
