import numpy as np


def at_least_2d(inputs):
    """ Checks if a tensor is less than 2-dimensional. If that's the case, it returns a 2D view of it, otherwise it returns the unchanged tensor. """

    if len(inputs.size()) < 2:
        return inputs.view((-1, 1))
    else:
        return inputs


def quartile(t, q, dim):
    """
    Return the ``q``-th quartile of the flattened input tensor's data.

    From https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """

    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.

    if dim is None:
        k = 1 + round(float(q) * (t.numel() - 1))
        if k < 1:
            k = 1
        elif dim is None and k > t.numel():
            k = t.numel()
        result = t.view(-1).kthvalue(k).values.item()
    else:
        k = 1 + round(float(q) * (t.size(dim) - 1))
        if k < 1:
            k = 1
        elif k > t.size(dim):
            k = t.size(dim)
        result = t.kthvalue(k, dim=dim).values

    return result


def nats_to_bits_per_dim(nats, dimensions):
    """ Converts log likelihoods in nats to log likelihoods in bpd. """
    return nats / (np.log(2) * np.prod(dimensions))
