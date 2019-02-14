#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Analysis functions.

@author: enricobagli
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import analyse
from . import hist_func
from . import harmonic
from . import pdf
from . import fit
from matplotlib.colors import LogNorm


'''
Plot Beam Properties
'''

def incoming_divergence(ang_in_x_urad,
                        ang_in_y_urad,
                        bins=np.arange(-200,200,10)):
    
    def single(defl_series,
               ax,
               bins=np.arange(-200,200,10)):
        
        
    
        try:    
            he, bee = np.histogram(defl_series,
                                   bins = bins)
            
            centers = hist_func.get_centers_from_edges(bee)
            he[he==0.] = 1
            y_errors = np.sqrt(he)
            y_errors[y_errors==0] = 1.
            x_errors = hist_func.get_bin_size_array_from_edges(bee) 
            
            output = fit.with_errors(
                fit_func = pdf.gaussian,
                xs       = centers,
                xs_err   = x_errors,
                ys       = he,
                ys_err   = y_errors,
                p0       = [np.max(he),
                            0.,
                            np.std(ang_in_x_urad)])
            
            fit_p = output.beta
            fit_e = output.sd_beta        
    
            hist, bin_edges = hist_func.normalize(output.y, bee)
            ax.plot(centers,
                       hist)
            str_ang = '\nRMS [urad] = %.2f $\pm$ %.2f'%(fit_p[2],fit_e[2])
            ax.text(0.05,
                    0.95,
                    str_ang,
                    fontsize = 12,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
            return fit_p, fit_e
        except:
            return None, None
            pass
        
    fig, ax = plt.subplots(figsize=(16,5),
                           ncols=2,
                           sharex=True,
                           sharey=True)

    ax[0].hist(
            ang_in_x_urad,
            bins=bins,
            alpha=0.7,
            normed=True,
            label='Horizontal')
    ax[1].hist(
            ang_in_y_urad,
            bins=bins,
            alpha=0.7,
            normed=True,
            label='Vertical')
    
    fit_px, fit_ex = single(
            defl_series=ang_in_x_urad,
            ax=ax[0],
            bins=bins)
    
    fit_py, fit_ey = single(
            defl_series=ang_in_y_urad,
            ax=ax[1],
            bins=bins)
    ax[0].set_title('Horizontal')
    ax[1].set_title('Vertical')
    return fit_px, fit_ex, fit_py, fit_ey, fig, ax
    
def deflection_vs_position(
        position_x_series,
        position_y_series,
        deflection_series,
        pos_x_cut_1,
        pos_x_cut_2,
        pos_y_cut_1,
        pos_y_cut_2,
        bins_position_x = np.arange(-5,5.,0.05),
        bins_position_y = np.arange(-5,5.,0.05),
        bins_deflection = np.arange(-100.,100.,1.)):
    fig, ax = plt.subplots(figsize=(16,5),ncols=2,sharey=True)
 
    
    pos_x_centers = hist_func.get_centers_from_edges(bins_position_x)
    pos_group = deflection_series.groupby(pd.cut(position_x_series,
                                          bins_position_x)).std()
    try:
        threshold_line = np.max(pos_group.values)*0.7 - np.min(pos_group.values)
        mask = pos_group.values > threshold_line
        
        xmin = np.min(pos_x_centers[mask])
        xmax = np.max(pos_x_centers[mask])
    except:
        xmin = bins_position_x[0]
        xmax = bins_position_x[-1]
        
    mask_posx = hist_func.get_selection_mask(
        position_x_series,
        selection_width = hist_func.get_width_x12(pos_x_cut_1,
                                                  pos_x_cut_2),
        selection_mean  = hist_func.get_mean_x12(pos_x_cut_1,
                                                 pos_x_cut_2))

    mask_posy = hist_func.get_selection_mask(
        position_y_series,
        selection_width = hist_func.get_width_x12(pos_y_cut_1,
                                                  pos_y_cut_2),
        selection_mean  = hist_func.get_mean_x12(pos_y_cut_1,
                                                 pos_y_cut_2))

    ax[0].hist2d(
        position_x_series[mask_posy],
        deflection_series[mask_posy],
        bins=(bins_position_x,
              bins_deflection),
        norm=LogNorm()
    )
    ax[0].set_xlabel('Horizontal Position [mm]')
    ax[0].set_ylabel('Outgoin Angle [urad]')
    ax[0].axvline(pos_x_cut_1,color='red')
    ax[0].axvline(pos_x_cut_2,color='red')
    ax[0].axvline(xmin,color='blue')
    ax[0].axvline(xmax,color='blue')


    ax[1].hist2d(
        position_y_series[mask_posx],
        deflection_series[mask_posx],
        bins=(bins_position_y,
              bins_deflection),
        norm=LogNorm()
    )
    ax[1].set_xlabel('Vertical Position [mm]')
    ax[1].axvline(pos_y_cut_1,color='red')
    ax[1].axvline(pos_y_cut_2,color='red')
    
    return mask_posx, mask_posy, fig, ax
    
def efficiency_with_gaussian(
    x_bins,
    x_width,
    eff_val,
    eff_err,
    eff_cut,
    p0,
    plot     = False,
    xlabel   ='Incoming Angle [$\mu$rad]',
    ax       = None):

    mask = eff_val>eff_cut
    
    ia_fit = x_bins[mask]
    ee_fit = eff_err[mask]
    ev_fit = eff_val[mask]

    output = fit.efficiency_with_gaussian(
        x_bins,
        x_width,
        eff_val,
        eff_err,
        eff_cut,
        p0)
    
    if plot is True:
        if ax == None:
            fig,ax = plt.subplots(figsize=(8,8))

        ax.errorbar(ia_fit,
                    ev_fit,
                    ls = "--",
                    color = 'black',
                    xerr=x_width,
                    yerr=ee_fit)
        ax.plot(ia_fit,
                output.y,
                ls = "--",
                color = 'red')

        ax.set_xlim(x_bins[0],x_bins[-1])
        ax.set_ylim(0.,100.)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Efficiency [%]')
        ax.grid()

        str_const    = r'$A_{fit}$ [%]' +' = %.3f $\pm$ %.3f\n'%(
            output.beta[0],
            output.sd_beta[0])
        str_mean    = '$\mu_{fit}$ [${\mu}$rad] = %.3f $\pm$ %.3f\n'%(
            output.beta[1],
            output.sd_beta[1])
        str_sigma   = '$\sigma_{fit}$ [${\mu}$rad] = %.3f $\pm$ %.3f'%(
            output.beta[2],
            output.sd_beta[2])

        ax.text(0.05,
                0.95,
                str_const+str_mean+str_sigma,
                fontsize = 12,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)

        return output, ax
    else:
        return output


def curvature_variation(
        pos_in_ys,
        pos_in_y_step,
        fit_v,
        fit_e, 
        ang_v, 
        ang_e,
        fit_func = pdf.linear):
    fig,ax = plt.subplots(
            figsize = (8,14),
            nrows   = 3,
            sharex  = True)

    def plot_linear_fit(ax,
                        xlabel,
                        ys,
                        ys_err,
                        fit_func = pdf.linear,
                        p0      = None):
        ax.errorbar(
            pos_in_ys,
            ys,
            color = 'black',
            xerr  = pos_in_y_step,
            yerr  = ys_err,
            alpha = 0.5)
    
        if p0 is None and fit_func == pdf.linear:
            p0 = [np.mean(ys),0.]
        if p0 is None and fit_func == pdf.quadratic:
            p0 = [np.mean(ys),0.,0.]
            
            
        output = fit.with_errors(
            fit_func = fit_func,
            xs       = pos_in_ys,
            xs_err   = pos_in_y_step,
            ys       = ys,
            ys_err   = ys_err,
            p0       = p0)
    
        ax.plot(
            pos_in_ys,
            output.y,
            color ='red',
            ls    ='--',
            lw    = 2,
            alpha = 0.5
        )
        str_p = ''
        for i in range(len(output.beta)):
            str_p+='p%d = %.2f $\pm$ %.2f\n' %(i,
                                               output.beta[i],
                                               output.sd_beta[i])
        ax.text(0.05,
                0.95,
                str_p,
                fontsize = 12,
                color = 'red',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)
    
        ax.grid()
        ax.set_xlim(pos_in_ys[0],pos_in_ys[-1])
        ax.set_ylabel(xlabel)
    
    
    plot_linear_fit(ax     = ax[0],
                    xlabel = 'Channeling Zero Angle [$\mu$rad]',
                    ys     = fit_v[:,1],
                    ys_err = fit_e[:,1],
                    fit_func = fit_func)
    ax[0].set_ylim(-30.,30.)

    
    plot_linear_fit(ax     = ax[1],
                    xlabel = 'Maximum Efficiency [%]',
                    ys     = fit_v[:,0],
                    ys_err = fit_e[:,0],
                    fit_func = fit_func)
    ax[1].set_ylim(0.,100.)


    plot_linear_fit(ax     = ax[2],
                    xlabel = 'Channeling Deflection Angle [$\mu$rad]',
                    ys     = ang_v,
                    ys_err = ang_e,
                    fit_func = fit_func)
    ax[2].set_ylim(np.mean(ang_v) - 30.,
                   np.mean(ang_v) + 30.)
    ax[2].set_ylim(40.,
                   60.)
    ax[2].set_xlabel('Vertical Position [mm]')
    fig.tight_layout()
    
    return fig, ax



channeling_efficiency_output_columns = [
        'particles',
        'ang_in_x_cut_urad',
       'half_eff',
       'ch_mean_urad',
       'ch_mean_err_urad',
       'ch_eff',
       'ch_eff_err',
       'ch_sigma_urad',
       'ch_sigma_err_urad']


def channeling_efficiency(
        ang_in_x_urad_series,
        defl_x_urad_series,
        cut_ang_x_in,
        sigma_cut_urad,
        half_efficiency,
        error_in_urad,
        torsion,
        torsion_err,
        incoming_angle_shift,
        incoming_angle_shift_err,
        fit_func,
        bins,
        p0,
		output_result = False):
    
    hist, bin_edges, deflection_series = analyse.get_channeling_hist(
        incoming_angle_series = ang_in_x_urad_series,
        deflection_series     = defl_x_urad_series,
        selection_width       = cut_ang_x_in,
        selection_mean        = 0.,
        bins                  = bins)

    centers = hist_func.get_centers_from_edges(bin_edges)

    he, bee = np.histogram(deflection_series,
                           bins = bins)
    
    he[he==0.] = 1
    y_errors = hist/np.sqrt(he)
    y_errors[y_errors==0] = 1.
    x_errors = hist_func.get_bin_size_array_from_edges(bee) 
    
    try:
        output = fit.with_errors(
            fit_func = fit_func,
            xs       = centers,
            xs_err   = x_errors,
            ys       = hist,
            ys_err   = y_errors,
            p0       = p0)
        fit_p = output.beta
        fit_e = output.sd_beta
        
        ch_m = fit_p[1]
        ch_s = fit_p[2]
    
        eff,eff_err = analyse.get_channeling_efficiency(
            deflection_series = deflection_series,
            selection_width   = sigma_cut_urad*ch_s,
            selection_mean    = ch_m,
            half_efficiency   = half_efficiency)*100.
    
        m_es, s_es, e_es = analyse.get_channeling_systematic_error(
            ang_in_x_urad_series = ang_in_x_urad_series,
            defl_x_urad_series   = defl_x_urad_series,
            cut_ang_x_in         = cut_ang_x_in,
            sigma_cut_urad       = sigma_cut_urad,
            half_efficiency      = half_efficiency,
            error_in_urad        = error_in_urad,
            channeling_mean      = ch_m,
            channeling_sigma     = ch_s,
            channeling_eff       = eff,
            p0                   = p0,
            bins                 = bins,
            fit_func             = fit_func)
    except:
        fit_e = np.zeros([3])
        mask1 = defl_x_urad_series > (p0[1] - sigma_cut_urad*p0[2])
        mask2 = defl_x_urad_series < (p0[1] + sigma_cut_urad*p0[2])
        ch_m  = np.mean(defl_x_urad_series[mask1&mask2])
        ch_s  = np.std(defl_x_urad_series[mask1&mask2])
        m_es  = 0.
        s_es  = 0.
        e_es  = 0.
        
        eff,eff_err = analyse.get_channeling_efficiency(
            deflection_series = deflection_series,
            selection_width   = sigma_cut_urad*ch_s,
            selection_mean    = ch_m,
            half_efficiency   = half_efficiency)*100.

    '''
    Plot
    '''

    fig,ax = plt.subplots(figsize=(7,4))

    centers = hist_func.get_centers_from_edges(bin_edges)
    
    ax.plot(
        centers,
        hist,
        color='black')

    try:
        fit_y = fit_func(
            fit_p,
            centers)
        
        ax.plot(
            centers,
            fit_y,
            '--',
            color='red')
    except:
        pass

    if half_efficiency is True:
        line1 = ch_m
        line2 = ch_m+sigma_cut_urad*ch_s
    else:
        line1 = ch_m-sigma_cut_urad*ch_s
        line2 = ch_m+sigma_cut_urad*ch_s

    ax.axvline(
        line1,
        ls='--')
    ax.axvline(
        line2,
        ls='--')

    bins_cut = np.where((centers>line1)&(centers<line2))
    ax.fill_between(
        x  = centers[bins_cut],
        y1 = hist[bins_cut],
        alpha = 0.3)
    
    ax.set_xlabel('Deflection Angle [urad]')
    ax.set_ylabel('Normalized Counts [a.u.]')

    str_ang_cut = r'${\theta}_{cut}$' +  ' [${\mu}$rad] = %.1f $\pm$ %.1f\n'%(
        cut_ang_x_in,
        error_in_urad)
    str_mean    = '$\mu_{ch}$ [urad] = %.1f $\pm$ %.1f\n'%(
        ch_m,
        m_es+fit_e[1])
    str_eff     = '$\epsilon_{ch}$ [%]' +' = %.1f $\pm$ %.1f\n'%(
        eff,
        e_es+eff_err)
    str_sigma   = '$\sigma_{ch}$ [${\mu}$rad] = %.2f $\pm$ %.2f'%(
        ch_s,
        s_es+fit_e[2])
    str_torsion = r'${\tau}$'+' [${\mu}$rad/mm] = %.1f $\pm$ %.1f\n' %(
        torsion,
        torsion_err)
    str_angin = r'${\theta}_{in,0}$'+' [${\mu}$rad] = %.1f $\pm$ %.1f\n' %(
        incoming_angle_shift,
        incoming_angle_shift_err)
    
    ax.set_xlim(bins[0],
                bins[-1])
    ax.set_ylim(0.,
                0.028)
    
    ax.text(0.05,
            0.95,
            str_angin+str_ang_cut+str_torsion+str_mean+str_eff+str_sigma,
            fontsize = 12,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)

    result = np.zeros(9)
    result[0] = deflection_series.shape[0]
    result[1] = int(cut_ang_x_in)
    result[2] = int(half_efficiency)
    result[3] = ch_m
    result[4] = m_es+fit_e[1]
    result[5] = eff
    result[6] = e_es+eff_err
    result[7] = ch_s
    result[8] = s_es+fit_e[2]
    
    if output_result:
	    return result, fig, ax, fit_p, fit_e
    
    return result, fig, ax
    
def channeling_deflection_distributions(
        ang_in_x_urad_series,
        defl_x_urad_series,
        cut_ang_x_ins = [16,
                         8,
                         4],
        colors        = ['black',
                         'green',
                         'red'],
        bins          = np.arange(-100,102,2)):
    fig,ax = plt.subplots(figsize=(7,4))
    
    for a,c in zip(cut_ang_x_ins,colors):
        hist, bin_edges, deflection_series = analyse.get_channeling_hist(
            incoming_angle_series = ang_in_x_urad_series,
            deflection_series     = defl_x_urad_series,
            selection_width       = a,
            selection_mean        = 0.,
            bins                  = bins)

        centers = hist_func.get_centers_from_edges(bin_edges)

        ax.plot(
            centers,
            hist,
            color = c,
            label = 'cut $\pm$ %d $\mu$rad'%(int(a)))
        
    ax.set_xlim(bins[0],bins[-1])
    ax.set_ylim(0.,0.025)
    ax.grid()
    ax.set_xlabel('Deflection Angle [urad]')
    ax.set_ylabel('Normalized Counts [a.u.]')
    ax.legend(loc=2,frameon=False)

    return fig, ax


import collections
def theoretical_efficiency(
    incoming_angles_urad,
    incoming_angle_resolution_urad = 5.4,
    beam_momentum_GeV = 180,
    beam_particle_mass_GeV = 0.13957018,
    beam_charge = +1.,
    crystal_length_m = 4.E-3,
    crystal_bending_radius_m = 80,
    crystal_el_field_max_Gev_on_cm = 5.7,
    crystal_pot_depth_eV = 16.,
    crystal_interplanar_distance_A = 1.92,
    crystal_Z = 14.,
    crystal_ionization_energy_eV = 172.,
    thermal_vibration_amplitude_A  = 0.075,
    thermal_vibration_rms_number   = 2.5,
    ax       = None) :
    
    if isinstance(crystal_pot_depth_eV, collections.Iterable):
        for ef,pot,ia in zip(crystal_el_field_max_Gev_on_cm,
                             crystal_pot_depth_eV,
                             crystal_interplanar_distance_A):
            harmonic.get_channeling_parameter_table(
                beam_momentum_GeV,
                0.,
                beam_particle_mass_GeV,
                beam_charge,
                crystal_length_m,
                crystal_bending_radius_m,
                ef,
                pot,
                ia,
                crystal_Z,
                crystal_ionization_energy_eV,
                thermal_vibration_amplitude_A,
                thermal_vibration_rms_number)
            print('\n')
    
    else:
        harmonic.get_channeling_parameter_table(
            beam_momentum_GeV,
            0.,
            beam_particle_mass_GeV,
            beam_charge,
            crystal_length_m,
            crystal_bending_radius_m,
            crystal_el_field_max_Gev_on_cm,
            crystal_pot_depth_eV,
            crystal_interplanar_distance_A,
            crystal_Z,
            crystal_ionization_energy_eV,
            thermal_vibration_amplitude_A,
            thermal_vibration_rms_number)

    effs = np.zeros(incoming_angles_urad.shape)
    if isinstance(crystal_pot_depth_eV, collections.Iterable):
        ia_tot = np.sum(crystal_interplanar_distance_A)
        for ef,pot,ia in zip(crystal_el_field_max_Gev_on_cm,
                             crystal_pot_depth_eV,
                             crystal_interplanar_distance_A):
            for i,a in enumerate(incoming_angles_urad):
                effs[i] += harmonic.calc_channeling_efficiency(
                    beam_momentum_GeV,
                    a*1.E-3,
                    beam_particle_mass_GeV,
                    beam_charge,
                    crystal_length_m,
                    crystal_bending_radius_m,
                    ef,
                    pot,
                    ia,
                    crystal_Z,
                    crystal_ionization_energy_eV,
                    thermal_vibration_amplitude_A,
                    thermal_vibration_rms_number)*100.*ia/ia_tot
            
    else:
        for i,a in enumerate(incoming_angles_urad):
            effs[i] = harmonic.calc_channeling_efficiency(
                beam_momentum_GeV,
                a*1.E-3,
                beam_particle_mass_GeV,
                beam_charge,
                crystal_length_m,
                crystal_bending_radius_m,
                crystal_el_field_max_Gev_on_cm,
                crystal_pot_depth_eV,
                crystal_interplanar_distance_A,
                crystal_Z,
                crystal_ionization_energy_eV,
                thermal_vibration_amplitude_A,
                thermal_vibration_rms_number)*100.
    
    bin_edges = hist_func.get_edges_from_centers(incoming_angles_urad)
    hist, bin_edges = hist_func.convolve_gaussian(
            effs,
            bin_edges,
            gaus_sigma = incoming_angle_resolution_urad)
    
    efficiency_with_gaussian(
        x_bins  = incoming_angles_urad,
        x_width = incoming_angles_urad[1] - incoming_angles_urad[0],
        eff_val = hist,
        eff_err = np.ones(incoming_angles_urad.shape) / 100.,
        eff_cut = 0.,
        p0      = [np.max(effs),
                   np.mean(incoming_angles_urad),
                   np.std(incoming_angles_urad)],
        plot     = True,
        xlabel   ='Incoming Angle [$\mu$rad]',
        ax       = ax)
    return hist