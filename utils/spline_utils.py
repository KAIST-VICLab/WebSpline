import torch


def interpolate_cubic_hermite(signal, times, N):
    """Cubic Hermite interpolation of per-Gaussian control points.

    signal: (P, 3, C) control values, padded to C per Gaussian
    times:  (P, 3, 1) query time in [0, 1]
    N:      (P, 1) number of control points actually in use per Gaussian
    returns (P, 3)

    Derivatives are one-sided at the two boundary segments and centred
    elsewhere; indices are clamped so the padding past N is never read.
    """
    times_scaled = times * (N - 1)[:, None]
    indices = torch.floor(times_scaled).long()

    lo = torch.zeros_like(N)[:, None].expand(-1, 3, -1)
    hi_seg = (N - 2)[:, None].expand(-1, 3, -1)
    hi = (N - 1)[:, None].expand(-1, 3, -1)
    indices = torch.clamp(indices, lo, hi_seg).long()
    left_indices = torch.clamp(indices - 1, lo, hi).long()
    right_indices = torch.clamp(indices + 1, lo, hi).long()
    right_right_indices = torch.clamp(indices + 2, lo, hi).long()

    t = times_scaled - indices.float()
    p0 = torch.gather(signal, -1, left_indices)
    p1 = torch.gather(signal, -1, indices)
    p2 = torch.gather(signal, -1, right_indices)
    p3 = torch.gather(signal, -1, right_right_indices)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t ** 2 * (3 - 2 * t)
    h11 = t ** 2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
    return interpolation.squeeze(-1)
