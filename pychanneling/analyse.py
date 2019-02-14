#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Analysis functions.

@author: enricobagli
"""
import numpy as np

from . import hist_func
from . import pdf
from . import fit

'''
Support Functions
'''

def get_channeling_efficiency(deflection_series,
                              selection_width,
                              selection_mean,
                              half_efficiency = False):
    mask_ch = hist_func.get_selection_mask(deflection_series,
                               selection_width = selection_width,
                               selection_mean  = selection_mean)
    if half_efficiency is True:
        mask_ch = hist_func.get_selection_mask(
            deflection_series,
            selection_width = selection_width*0.5,
            selection_mean  = selection_mean+selection_width*0.5)

    counts_ch  = np.sum(mask_ch)
    counts_tot = deflection_series.count()
    if half_efficiency is True:
        counts_ch = counts_ch*2.
    
    eff = 0.
    err = 0.
    if counts_tot > 0.:
        eff = counts_ch/counts_tot
        if counts_ch > 0.:
            err = eff*np.sqrt((counts_ch+counts_tot)/(counts_ch*counts_tot))
        else:
            err = 1./np.sqrt(counts_tot)

    return np.array([eff,err])
    
def get_channeling_hist(incoming_angle_series,
                        deflection_series,
                        selection_width,
                        selection_mean,
                        bins = np.arange(-100,100,1)):
    mask = hist_func.get_selection_mask(
            incoming_angle_series,
            selection_width = selection_width,
            selection_mean  = selection_mean)

    hist, bin_edges = np.histogram(deflection_series[mask],
                                   bins = bins)

    hist, bin_edges = hist_func.normalize(hist, bin_edges)
    return hist, bin_edges, deflection_series[mask]


def get_channeling_systematic_error(
    ang_in_x_urad_series,
    defl_x_urad_series,
    cut_ang_x_in,
    sigma_cut_urad,
    half_efficiency,
    error_in_urad,
    channeling_mean,
    channeling_sigma,
    channeling_eff,
    p0,
    bins,
    fit_func):
    
    m_es = 0
    s_es = 0
    e_es = 0
    
    for error in [-error_in_urad,+error_in_urad]:
        h, be, ds = get_channeling_hist(
            incoming_angle_series = ang_in_x_urad_series,
            deflection_series     = defl_x_urad_series,
            selection_width       = cut_ang_x_in,
            selection_mean        = error,
            bins                  = bins)

        centers = hist_func.get_centers_from_edges(be)
        
        he, bee = np.histogram(defl_x_urad_series,
                               bins = bins)
        
        he[he==0.] = 1
        y_errors = h/np.sqrt(he)
        y_errors[y_errors==0] = 1.
        x_errors = hist_func.get_bin_size_array_from_edges(bee) 
        
        output = fit.with_errors(
            fit_func = fit_func,
            xs       = centers,
            xs_err   = x_errors,
            ys       = h,
            ys_err   = y_errors,
            p0       = p0)
        fit_p = output.beta
        fit_e = output.sd_beta        
        
        ev,er = get_channeling_efficiency(
            deflection_series = ds,
            selection_width   = sigma_cut_urad*fit_p[2],
            selection_mean    = fit_p[1],
            half_efficiency   = half_efficiency)*100.

        m_es += np.square(fit_p[1] - channeling_mean) 
        m_es += np.square(fit_e[1])
        
        s_es += np.square(fit_p[2] - channeling_sigma) 
        s_es += np.square(fit_e[2]) 

        e_es += np.square(ev - channeling_eff) 
        e_es += np.square(er) 
        
    return np.sqrt(m_es), np.sqrt(s_es), np.sqrt(e_es)


'''
Calculate Efficiency Vs Incoming Angle
'''

def efficiency_vs_incoming_angle(
    incoming_angles,
    incoming_angle_series,
    deflection_series,
    selection_width = 2,
    bins            = np.arange(-100,100,1),
    s_cut_urad      = 3.,
    half_eff        = False,
    fit_func        = pdf.gaussian,
    p0              = None
    ):
    
    e_vals = np.zeros(incoming_angles.shape)
    e_errs = np.zeros(incoming_angles.shape)
    popt_series = np.zeros([incoming_angles.shape[0],3])
    pcov_series = np.zeros([incoming_angles.shape[0],3])
    
    for i,ang in enumerate(incoming_angles):
        hist, bin_edges, deflection_series_t = get_channeling_hist(
            incoming_angle_series = incoming_angle_series,
            deflection_series = deflection_series,
            selection_width = selection_width,
            selection_mean = ang,
            bins = bins)
        try:
            centers = hist_func.get_centers_from_edges(bin_edges)

            he, bee = np.histogram(deflection_series_t,
                                   bins = bins)
            he[he==0.] = 1
           
            y_errors = hist/np.sqrt(he)
            y_errors[y_errors==0] = 1.
            x_errors = hist_func.get_bin_size_array_from_edges(bee) 
            
            output = fit.with_errors(
                fit_func = fit_func,
                xs       = centers,
                xs_err   = x_errors,
                ys       = hist,
                ys_err   = y_errors,
                p0       = p0)
            fit_p = output.beta
            fit_e = output.sd_beta
            
            mean = fit_p[1]
            sigma = fit_p[2]

            popt_series[i] = fit_p[:3]
            pcov_series[i] = fit_e[:3]
        except:
            popt_series[i] = np.zeros([3])
            pcov_series[i] = np.zeros([3])
            mean  = p0[1]
            sigma = p0[2]
            
        try:
            if half_eff == True:
                eff, eff_err = get_channeling_efficiency(
                    deflection_series_t,
                    selection_width = 0.5*(s_cut_urad*sigma),
                    selection_mean  = 0.5*(s_cut_urad*sigma) + mean)*200.            
            else:
                eff, eff_err = get_channeling_efficiency(
                    deflection_series_t,
                    selection_width = s_cut_urad*sigma,
                    selection_mean  = mean)*100.

            e_vals[i] = eff
            e_errs[i] = eff_err
        except:
            pass
        
    return e_vals, e_errs, popt_series, pcov_series

def efficiency_vs_torsion(
    torsions,
    pos_in_y_mm_series,
    eff_cut,
    incoming_angles,
    incoming_angle_series,
    deflection_series,
    selection_width = 2,
    bins            = np.arange(-100,100,1),
    s_cut_urad      = 3.,
    half_eff        = False,
    fit_func        = pdf.gaussian,
    p0              = None,
    ):

    
    fit_v = np.zeros([torsions.shape[0],3]) 
    fit_e = np.zeros([torsions.shape[0],3]) 

    for j,torsion in enumerate(torsions):
        angin_t  = incoming_angle_series.copy()
        angin_t -= torsion * pos_in_y_mm_series
        
        try:
            eff_val, eff_err, popt_s, pcov_s = efficiency_vs_incoming_angle(
                incoming_angles       = incoming_angles,
                incoming_angle_series = angin_t,
                deflection_series     = deflection_series,
                selection_width       = selection_width,
                bins                  = bins,
                s_cut_urad            = s_cut_urad,
                half_eff              = half_eff,
                fit_func              = fit_func,
                p0                    = p0)
        except:
            pass
        try:
            output_angin = fit.efficiency_with_gaussian(
                x_bins  = incoming_angles,
                x_width = selection_width,
                eff_val = eff_val,
                eff_err = eff_err,
                eff_cut = eff_cut,
                p0      = [np.max(eff_val),
                           np.mean(incoming_angles),
                           np.std(incoming_angles)])
            fit_v[j] = output_angin.beta
            fit_e[j] = output_angin.sd_beta
        except:
            pass

    return fit_v, fit_e

def curvature_variation(
    pos_in_ys,
    pos_in_y_step,
    pos_in_y_mm_series,
    eff_cut,
    incoming_angles,
    incoming_angle_series,
    deflection_series,
    selection_width = 2,
    bins            = np.arange(-100,100,1),
    s_cut_urad      = 3.,
    half_eff        = False,
    fit_func        = pdf.gaussian,
    p0              = None
    ):
    
    fit_v = np.zeros([pos_in_ys.shape[0],3]) 
    fit_e = np.zeros([pos_in_ys.shape[0],3]) 
    
    ang_v = np.zeros(pos_in_ys.shape) 
    ang_e = np.zeros(pos_in_ys.shape) 
    
    for j,pos_in_y in enumerate(pos_in_ys):
        mask = hist_func.get_selection_mask(
            series          = pos_in_y_mm_series,
            selection_width = pos_in_y_step,
            selection_mean  = pos_in_y)

        incoming_angle_series_in = incoming_angle_series[mask]
        deflection_series_in     = deflection_series[mask]
        try:
            eff_val, eff_err, popt_s, pcov_s = efficiency_vs_incoming_angle(
                incoming_angles       = incoming_angles,
                incoming_angle_series = incoming_angle_series_in,
                deflection_series     = deflection_series_in,
                selection_width       = selection_width,
                bins                  = bins,
                s_cut_urad            = s_cut_urad,
                half_eff              = half_eff,
                fit_func              = fit_func,
                p0                    = p0)
        except:
            pass
        try:
            output_angin =  fit.efficiency_with_gaussian(
                x_bins  = incoming_angles,
                x_width = selection_width,
                eff_val = eff_val,
                eff_err = eff_err,
                eff_cut = eff_cut,
                p0      = [np.max(eff_val),
                           np.mean(incoming_angles),
                           np.std(incoming_angles)])

            fit_v[j] = output_angin.beta
            fit_e[j] = output_angin.sd_beta
        except:
            pass

        try:
            hist, bin_edges, ds = get_channeling_hist(
                incoming_angle_series = incoming_angle_series_in,
                deflection_series     = deflection_series_in,
                selection_width       = selection_width,
                selection_mean        = output_angin.beta[1],
                bins                  = bins)

            centers = hist_func.get_centers_from_edges(bin_edges)

            he, bee = np.histogram(ds,
                                   bins = bins)
            he[he==0.] = 1
            y_errors = hist/np.sqrt(he)
            y_errors[y_errors==0] = 1.
            x_errors = hist_func.get_bin_size_array_from_edges(bee) 
            
            output = fit.with_errors(
                fit_func = fit_func,
                xs       = centers,
                xs_err   = x_errors,
                ys       = hist,
                ys_err   = y_errors,
                p0       = p0)
            fit_pg = output.beta
            fit_eg = output.sd_beta

            ang_v[j] = fit_pg[1]
            ang_e[j] = fit_eg[1] 
        except:
            def reject_outliers(data,
                                m=3):
                return data[abs(data - np.mean(data)) < m * np.std(data)]
            mask_defl = np.fabs(deflection_series_in - p0[1]) < p0[2] 
            dx = reject_outliers(deflection_series_in[mask_defl],3)
            ang_v[j] = np.mean(dx)
            ang_e[j] = np.std(dx)
    return fit_v, fit_e, ang_v, ang_e