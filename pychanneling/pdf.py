#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    PDF for coheren physics effects.
    
@author: Enrico Bagli
"""

import numpy as np

def get_par_number(function):
    # copied from scipy.optimize.curve_fit
    # determine number of parameters by inspecting the function
    from scipy._lib._util import getargspec_no_self as _getargspec
    args, varargs, varkw, defaults = _getargspec(function)
    if len(args) < 2:
        raise ValueError("Unable to determine number of fit parameters.")
    n = len(args) - 1
    return n

def linear(i,x):
    return i[0] + i[1]*x

def gaussian(i,x):
    '''
    Description:
        Gaussian PDF
    '''
    return i[0] * np.exp(-np.square(x - i[1]) / (2 * np.square(i[2])))

def double_gaussian(i,x):
     '''
     Description:
         Gaussian Sum PDF
     '''
     return gaussian(i[0:3],x) + gaussian(i[3:6],x)
 
def exponential(i,x):
    '''
    Description:
        Simple dechanneling PDF, i.e. an exponential function
    '''
    return i[0] * np.exp(- x / i[1])


def amorphous(i,x):
    '''
    Description:
        Amorphous peak PDF, i.e. sum of two gaussian functions
        
    Reference:
        Channeling, volume reflection, and volume capture study of electrons
        in a bent silicon crystal
        Authors: T. N. Wistisen et al.
        http://dx.doi.org/10.1103/PhysRevAccelBeams.19.071001
    '''
    gaus_1 = gaussian([i[3]     , i[1], i[2]       ], x)
    gaus_2 = gaussian([1. - i[3], i[1], i[2] * i[4]], x)
    return i[0] * (gaus_1 + gaus_2)

def dechanneling_erf(i,x):
    '''
    Description:
        Dechanneling PDF, i.e. sum of two ERF functions convoluted with an
        exponential function.
        
    Reference:
        Channeling, volume reflection, and volume capture study of electrons
        in a bent silicon crystal
        Authors: T. N. Wistisen et al.
        http://dx.doi.org/10.1103/PhysRevAccelBeams.19.071001

    da = dechanneling_angle
    dc = dechanneling_constant
    
    m1 = amorphous_mean

    m2 = channeling_mean
    s2 = channeling_sigma
    '''
    from scipy import special
    dc = i[0]
    da = i[1]
    
    m1 = i[2]

    m2 = i[3]
    s2 = i[4]
    
    dt = x - s2 * s2 / da + 1.E-10
    erf_1 = special.erf( (m1 - dt) / (np.sqrt(np.pi) * s2) )
    erf_2 = special.erf( (m2 - dt) / (np.sqrt(np.pi) * s2) )
    expo  = - (x-m1)/da + 0.5 * np.square(s2/da)

    pdf = (erf_2 - erf_1) * np.exp(expo) * 0.5 * dc / da
    
    return pdf * (pdf > 0)

def dechanneling_erf_double(i,x):
    '''
    Description:
        Double dechanneling PDFs (electronic and nuclear dechanneling).
    '''  
    dce_i = i[0:5]
    dcn_i = i[0:5]
    
    dce_i[0] *= i[6]
    dcn_i[0] *= (1.-i[6])
    
    dcn_i[1] = i[5]
    
    dce = dechanneling_erf(dce_i,x)
    dcn = dechanneling_erf(dcn_i,x)

    return dce + dcn

        
class channeling_dechanneling():
    '''
    Description:
        Channeling PDF, i.e. a sum of amorphous, channeling PDFs and two
        dechanneling PDFs (electronic and nuclear dechanneling)
        
    Reference:
        Channeling, volume reflection, and volume capture study of electrons
        in a bent silicon crystal
        Authors: T. N. Wistisen et al.
        http://dx.doi.org/10.1103/PhysRevAccelBeams.19.071001
        
        Steering efficiency of a ultrarelativistic proton beam in a thin bent
        crystal
        Authors: E. Bagli et al.
        https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-014-2740-7
 
   '''
   
    def __init__(
            self,   
            amorphous_model,
            channeling_model,
            dechanneling_model):
        self.am = amorphous_model
        self.ch = channeling_model
        self.de = dechanneling_model
        
    def set_par_number(self,
                       am_i,
                       ch_i,
                       de_i):
        
        self.am_i = am_i
        self.ch_i = ch_i
        self.de_i = de_i
    
    def model(self,
              i,
              x):
        am_p = [i[x] for x in self.am_i]
        ch_p = [i[x] for x in self.ch_i]
        de_p = [i[x] for x in self.de_i]
        
        am_f = self.am(am_p,x)
        ch_f = self.ch(ch_p,x)
        de_f = self.de(de_p,x)
        return am_f + ch_f + de_f
