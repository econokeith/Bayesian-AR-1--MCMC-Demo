from __future__ import division
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


def make_test_arma(ar_p=[], ma_q=[], sig=1,c=0, start=0, N=100, dists=None):
    """

    Makes Test ARMA(p,q) data

    :param ar_p: can either be float or list.
    :param ma_q: can either be float or list.
    :param sig: standard dev of error
    :param c: intercept
    :param start: starting value
    :param N: Length of series
    :param dists: can include disturbances.
    :return:
    """


    if isinstance(ar_p, float) or isinstance(ar_p, int):
        ar_p = [ar_p]

    if isinstance(ma_q, float) or isinstance(ma_q, int):
        ma_q = [ma_q]

    l_ar = len(ar_p)
    l_ma = len(ma_q)


    if dists:
        errors=dists
    else:
        errors = np.random.randn(N+l_ma)*sig

    arma = np.zeros(l_ar + N)
    arma[:l_ar] = start

    for i in xrange(len(errors) - l_ma):
        z = np.inner(arma[i:i + l_ar], ar_p) + np.inner(errors[i: i + l_ma], ma_q) + errors[i + l_ma]
        arma[i + l_ar] = z

    return arma[l_ar:]



def calc_q_star(data, phi):
    """
    Calculates Q* given phi
    For gibbs step for variance.
    :param data:
    :param phi:
    :return:
    """
    q1 = data[0] ** 2 * (1 - phi ** 2)
    qs = (data[1:]- phi * data[:-1])**2

    return q1+ qs.sum()


def fit_t_dist(data, plot=False, x_range=None, num=100,label=""):
    """
    Fits data to t distribution. If plot is True it will return a plot. Otherwise it will return
    a scipy function.
    :param data: numpy array
    :param plot: Bool
    :param xrange: If none it will automatically find range. otherwise, you can set it. must be tuple: (1,2)
    :param num: the number of points to plot i.e. num from numpy.linspace
    :return:
    """
    t_dist = sps.t.fit(data)
    t_dist = sps.t(*t_dist)

    if plot is False:
        return t_dist

    else:
        if x_range is None:
            x_min = data.min()
            x_max = data.max()

        else:
            x_min = x_range[0]
            x_max = x_range[1]

        xs = np.linspace(x_min, x_max, num=num)
        plt.plot(xs, t_dist.pdf(xs),label=label)

def fit_invgamma_dist(data, plot=False, x_range=None, num=100,label=""):
    """
    Fits data to inverse gamma distribution. If plot is True it will return a plot. Otherwise it will return
    a scipy function.
    :param data: numpy array
    :param plot: Bool
    :param xrange: If none it will automatically find range. otherwise, you can set it. must be tuple: (1,2)
    :param num: the number of points to plot i.e. num from numpy.linspace
    :return:
    """
    ig_dist = sps.invgamma.fit(data, floc=0)
    ig_dist = sps.invgamma(*ig_dist)

    if plot is False:
        return ig_dist

    else:
        if x_range is None:
            x_min = data.min()
            x_max = data.max()

        else:
            x_min = x_range[0]
            x_max = x_range[1]

        xs = np.linspace(x_min, x_max, num=num)
        plt.plot(xs, ig_dist.pdf(xs),label=label)






