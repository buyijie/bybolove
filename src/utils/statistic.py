#!/usr/bin/env python
# coding=utf-8

import math 

def Average(a) :
    """
    """
    return sum(a) * 1.0 / len(a)

def Covariance(a, b) :
    """
    """
    assert len(a) == len(b)
    ave_a = Average(a)
    ave_b = Average(b)
    return sum([(a[i] - ave_a) * (b[i] - ave_b) for i in xrange(len(a))]) / len(a)

def Variance(a) :
    """
    """
    ave = Average(a)
    return sum([(ai - ave) ** 2 for ai in a]) / len(a)

def CorrelationCoefficient(a, b) :
    """
    """
    return Covariance(a, b) / math.sqrt(Variance(a) * Variance(b))


