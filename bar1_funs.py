from __future__ import division
import numpy as np
import Quandl as qd
import pandas as pd
import scipy as sp
import numpy.random as npr
import scipy.stats as sps
from numpy import array
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from scipy import stats

import copy

import numpy.linalg as la
import pandas.io.data as web
from datetime import datetime


def make_test_arma(ar_p, ma_q, dev=100):
    """
    Makes Test ARMA(p,q) data
    :param ar_p:
    :param ma_q:
    :param dev:
    :return:
    """
    if isinstance(dev, int):
        errors = np.random.randn(dev)
    else:
        errors = dev

    l_ar = len(ar_p)
    l_ma = len(ma_q)

    z_arma_pq = np.empty(l_ar)
    z_arma_pq.fill(0)

    for i in xrange(len(errors) - l_ma):
        z = np.inner(z_arma_pq[i:i + l_ar], ar_p) + np.inner(errors[i: i + l_ma], ma_q) + errors[i + l_ma]
        z_arma_pq = np.append(z_arma_pq, z)

    return z_arma_pq


def q_star_ar1_phi(data, phi):
    """
    Calculates Q* given phi
    For gibbs step for variance.
    :param data:
    :param phi:
    :return:
    """
    q1 = data[0] ** 2 * (1 - phi ** 2)
    for i in xrange(1, len(data)):
        q1 += (data[i - 1] - phi * data[i])**2
    return q1

def ar1_mcmc(data, phi0, v0, iterations, c=2):

    data_len = len(data)
    v_list = [v0]
    phi_list = [phi0]
    for i in xrange(iterations):

        v = stats.invgamma.rvs(data_len/2, scale=q_star_ar1_phi(data, phi_list[-1]))
        v_list.append(v)

        phi_new = np.random.randn(1)[0]* c * v_list[-1] + phi_list[-1]
        phi_list.append(phi_new)

    return v_list, phi_list

def calc_likelihood(data, phi, v):
    """
    calclates the likelihood for hte MC step
    :param data:
    :param phi:
    :param v:
    :return:
    """
    n = len(data)
    prior = 1/v
    norm_constant = (1-phi**2)**.5 / (2*np.pi*v)**(n/2)
    q_star = q_star_ar1_phi(data, phi)
    return prior * norm_constant * np.exp(-q_star / 2*v)


def fit_t_dist(data, plot=False, x_range=None, num=100):
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
        plt.plot(xs, t_dist.pdf(xs))

def fit_invgamma_dist(data, plot=False, x_range=None, num=100):
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
        plt.plot(xs, ig_dist.pdf(xs))






