#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
@author: Enrico Bagli
"""

import numpy as np

from . import pdf
import scipy.odr as odr

def with_errors(
        fit_func,
        xs,
        xs_err,
        ys,
        ys_err,
        p0,
        fix = None):
    
    data = odr.RealData(
            x   = xs,
            y   = ys,
            sx  = xs_err,
            sy  = ys_err)
    
    model = odr.Model(fit_func)

    odr_instance = odr.ODR(data  = data,
                           model = model,
                           beta0 = p0)
    odr_instance.set_job(fit_type=2)
    
    return odr_instance.run()


def efficiency_with_gaussian(
    x_bins,
    x_width,
    eff_val,
    eff_err,
    eff_cut,
    p0):

    mask = eff_val>eff_cut
    
    ia_fit = x_bins[mask]
    ee_fit = eff_err[mask]
    ev_fit = eff_val[mask]


    output = with_errors(
        fit_func = pdf.gaussian,
        xs       = ia_fit,
        xs_err   = x_width,
        ys       = ev_fit,
        ys_err   = ee_fit,
        p0       = p0)
    
    return output