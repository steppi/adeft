import logging
from multiprocessing import Pool

import numpy as np
import scipy as sp
from scipy.stats import beta
from scipy.optimize import brentq
from scipy.special import loggama


logger = logging.getLogger(__file__)


def true_positives(y_true, y_pred, pos_label=1):
    return sum(1 if expected == pos_label and predicted == pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def true_negatives(y_true, y_pred, pos_label=1):
    return sum(1 if expected != pos_label and predicted != pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def false_positives(y_true, y_pred, pos_label=1):
    return sum(1 if expected != pos_label and predicted == pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def false_negatives(y_true, y_pred, pos_label=1):
    return sum(1 if expected == pos_label and predicted != pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def sensitivity_score(y_true, y_pred, pos_label=1):
    tp = true_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)
    try:
        result = tp/(tp + fn)
    except ZeroDivisionError:
        logger.warning('No positive examples in sample. Returning 0.0')
        result = 0.0
    return result


def specificity_score(y_true, y_pred, pos_label=1):
    tn = true_negatives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    try:
        result = tn/(tn + fp)
    except ZeroDivisionError:
        logger.warning('No negative examples in sample. Returning 0.0')
        result = 0.0
    return result


def youdens_j_score(y_true, y_pred, pos_label=1):
    sens = sensitivity_score(y_true, y_pred, pos_label)
    spec = specificity_score(y_true, y_pred, pos_label)
    return sens + spec - 1


def gamma_star(a):
    if a == 0:
        output = np.float('inf')
    elif a > 1e9:
        output = 1.0
    else:
        log_output = loggama(a) + a - 0.5*np.log(2*np.pi) - (a - 0.5)*np.log(a)
        output = np.exp(log_output)
    return output


def D(self, p, q, x):
    if x == 0 or x == 1:
        return 0.0
    part1 = np.sqrt(p*q/(2*np.pi * (p+q))) * \
        gamma_star(p+q)/(gamma_star(p)*gamma_star(q))
    x0 = p/(p+q)
    sigma = (x - x0)/x0
    tau = (x0 - x)/(1 - x0)
    part2 = np.exp(p*(np.log1p(sigma) - sigma) + q*(np.log1p(tau) - tau))
    return (part1 * part2)


def K(self, p, q, x, tol=1e-12):
    def coefficient(n):
        m = n // 2
        if n % 2 == 0:
            result = m*(q-m)/((p+2*m-1)*(p+2*m)) * x
        else:
            result = -(p+m)*(p+q+m)/((p+2*m)*(p+2*m+1)) * x
        return result
    delC = coefficient(1)
    C, D = 1 + delC, 1
    n = 2
    while abs(delC) > tol:
        D = 1/(D*coefficient(n) + 1)
        delC *= (D - 1)
        C += delC
        n += 1
    return 1/C


def betainc(self, p, q, x):
    if x > p/(p+q):
        return 1 - self._betainc(q, p, 1-x)
    else:
        return self._D(p, q, x)/p * self._K(p, q, x)


def prevalence_cdf(theta, n, t, sensitivity, specificity):
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    numerator = (betainc(t+1, n-t+1, c1 + c2*theta) - betainc(t+1, n-t+1, c1))
    denominator = betainc(t+1, n-t+1, c1 + c2) - betainc(t+1, n-t+1, c1)
    if denominator == 0:
        if t > n/2:
            return 1.0 if theta == 1.0 else 0.0
        elif t < n/2:
            return 1.0 if theta >= 0.5 else 0.0
    return numerator/denominator


def prevalence_credible_interval_exact(theta, n, t, sens, spec, alpha):
    def f(theta):
        return prevalence_cdf(theta, n, t, sens, spec)
    left = brentq(lambda x: f(x) - alpha/2, 0, 1, xtol=1e-3, rtol=1e-3,
                  maxiter=100)
    right = brentq(lambda x: f(x) - 1 + alpha/2, 0, 1, xtol=1e-3, rtol=1e-3,
                   maxiter=100)
    return (left, right)


def prevalence_credible_interval(theta, n, t, sens_shape, sens_range,
                                 spec_shape, spec_range, alpha, n_jobs=1,
                                 num_samples=5000):
    def sample_interval():
        sp.random.seed()
        return prevalence_credible_interval(n, t,
                                            beta.rvs(sens_shape[0],
                                                     sens_shape[1],
                                                     sens_range[0],
                                                     sens_range[1]),
                                            beta.rvs(spec_shape[0],
                                                     spec_shape[1],
                                                     spec_range[0],
                                                     spec_range[1]), alpha)
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            future_results = [pool.apply_async(sample_interval)
                              for i in range(num_samples)]
            results = [interval.get() for interval in future_results]
        return (np.mean([t[0] for t in results]),
                np.mean([t[1] for t in results]))
