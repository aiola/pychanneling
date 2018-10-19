#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Functions for the calculation of interesting quantities related to the
    channeling physics effect in bent crystal at high-energy.
    The harmonic approximation of the continuum potential is adopted.
    
@author: Enrico Bagli
"""

import numpy as np

def get_beta(beam_gamma):
    '''
    Description:
        Get beta from Lorentz's gamma
    
    Reference:
        https://en.wikipedia.org/wiki/Lorentz_factor
    '''
    return np.sqrt(1. - 1./np.square(beam_gamma))
    
def get_gamma(beam_energy_GeV,
              beam_particle_mass_GeV):
    '''
    Description:
        Get Lorentz's gamma
    
    Reference:
        https://en.wikipedia.org/wiki/Lorentz_factor
    '''
    return beam_energy_GeV/beam_particle_mass_GeV

def get_energy(beam_momentum_GeV,
               beam_particle_mass_GeV):
    '''
    Description:
        Get relativistic energy
    
    Reference:
        https://en.wikipedia.org/wiki/Energy%E2%80%93momentum_relation
    '''
    beam_p = beam_momentum_GeV
    beam_m = beam_particle_mass_GeV
    return np.sqrt(np.square(beam_p)+np.square(beam_m)) 
    
def get_deflection_angle_rad(crystal_length_m,
                              crystal_bending_radius_m):
    '''
    Description:
        Return the crystal deflection angle [mrad] from the length [mm] and the 
        bending radius [m]. The curvature is supposed to be uniform.
        
    Reference:
        Crystal Channeling and Its Application at High-Energy Accelerators
        Authors: Biryukov, Valery M., Chesnokov, Yuri A., Kotov, Vladilen I.
        https://www.springer.com/it/book/9783540607694
    '''
    return crystal_length_m/crystal_bending_radius_m

def get_Thomas_Fermi_radius_A(crystal_Z):
    '''
    Description:
    '''
    Bohr_radius = 0.529          # A   - Bohr radius
    return Bohr_radius * 0.88534 / np.cbrt(crystal_Z);

def get_bending_radius_m(crystal_length_m,
                         deflection_angle_rad):
    '''
    Description:
        Return the crystal bending radius [m] from the length [mm] and the 
        deflection angle [mrad]. The curvature is supposed to be uniform.

    Reference:
        Crystal Channeling and Its Application at High-Energy Accelerators
        Authors: Biryukov, Valery M., Chesnokov, Yuri A., Kotov, Vladilen I.
        https://www.springer.com/it/book/9783540607694
    '''
    return crystal_length_m/deflection_angle_rad

def get_critical_radius_m(momentum_velocity_GeV,
                          crystal_el_field_max_Gev_on_cm = 5.7):
    '''
    Description:
        Return the crystal critical bending radius [m] from the particle
        momentum-velocity [GeV] and the maximum of the crystal electric field
        [GeV/cm] under continuum potential approximation
   
    Reference:
        Crystal Channeling and Its Application at High-Energy Accelerators
        Authors: Biryukov, Valery M., Chesnokov, Yuri A., Kotov, Vladilen I.
        https://www.springer.com/it/book/9783540607694
    '''

    return momentum_velocity_GeV/crystal_el_field_max_Gev_on_cm*1.E-2

def get_critical_angle_mrad(momentum_velocity_GeV,
                            crystal_bending_radius_m = 1.E10,
                            crystal_pot_depth_eV = 16.,
                            crystal_el_field_max_Gev_on_cm = 5.7):
    '''
    Description:
        Return the crystal critical angle [mrad] from the particle
        momentum-velocity [GeV] and the crystal potential well depth [eV]
        under continuum potential approximation
    
    Reference:
        Crystal Channeling and Its Application at High-Energy Accelerators
        Authors: Biryukov, Valery M., Chesnokov, Yuri A., Kotov, Vladilen I.
        https://www.springer.com/it/book/9783540607694
    '''
    pb  = momentum_velocity_GeV
    cpd = crystal_pot_depth_eV * 1.E-9
    cbr = crystal_bending_radius_m
    cr  = get_critical_radius_m(pb,
                                crystal_el_field_max_Gev_on_cm)

    return np.sqrt(2. * cpd / pb) * 1.E3 * (1. - cr/cbr)

def calc_dechanneling_length_n_m(momentum_velocity_GeV):
    '''
    Description:
        Get the nuclear dechanneling length.
        
    Reference:
        Observation of nuclear dechanneling for high-energy protons in crystals
        Authors: W. Scandale et al.
        https://www.sciencedirect.com/science/article/pii/S0370269309010089

        Steering efficiency of a ultrarelativistic proton beam in a thin bent
        crystal
        Authors: E. Bagli et al.
        https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-014-2740-7
    '''
    return 0.0015 * np.sqrt(momentum_velocity_GeV / 400.) # m

def calc_dechanneling_length_e_m(beam_momentum_GeV,
                                 beam_particle_mass_GeV = 0.9382720813,
                                 crystal_bending_radius_m = 1.E10,
                                 crystal_el_field_max_Gev_on_cm = 5.7,
                                 crystal_interplanar_distance_A = 1.92,
                                 crystal_Z = 14.,
                                 crystal_ionization_energy_eV = 172.):
    '''
    Description:
        Get the electronic dechanneling length.
        
    Reference:
        Crystal Channeling and Its Application at High-Energy Accelerators
        Authors: Biryukov, Valery M., Chesnokov, Yuri A., Kotov, Vladilen I.
        https://www.springer.com/it/book/9783540607694
    '''

    e_mass      = 0.000510998910 # GeV - electron mass
    e_radius    = 2.8179E-5      # A   - electron radius
    TF_radius   = get_Thomas_Fermi_radius_A(crystal_Z)
    
    beam_p      = beam_momentum_GeV
    beam_m      = beam_particle_mass_GeV
    
    beam_energy = get_energy(beam_p,beam_m)
    beam_gamma  = get_gamma(beam_energy,beam_m)
    beam_beta   = get_beta(beam_gamma)

    cry_id      = crystal_interplanar_distance_A       # A
    cry_ie      = crystal_ionization_energy_eV * 1.E-9 # GeV

    critical_radius = get_critical_radius_m(beam_p*beam_beta,
                                            crystal_el_field_max_Gev_on_cm)

    dl  = 256./(9.*np.pi*np.pi)
    dl *= TF_radius*cry_id/(e_mass*e_radius)
    dl *= (beam_p*beam_beta/(np.log(2.*e_mass*beam_gamma/cry_ie)-1.)) * 1.E-10
    if crystal_bending_radius_m < critical_radius:
        return 0
    else:
        return dl * np.square(1. - critical_radius/crystal_bending_radius_m) 

def get_geometrical_acceptance(crystal_interplanar_distance_A = 1.92,
                               thermal_vibration_amplitude_A  = 0.075,
                               thermal_vibration_rms_number   = 2.5):
    '''
    Description:
        Get the channeling geometrical acceptance.
     
    Reference:
        Crystal Channeling and Its Application at High-Energy Accelerators
        Authors: Biryukov, Valery M., Chesnokov, Yuri A., Kotov, Vladilen I.
        https://www.springer.com/it/book/9783540607694
   '''
    tva     = thermal_vibration_amplitude_A  # A
    tva_rms = thermal_vibration_rms_number   # pure number
    cry_id  = crystal_interplanar_distance_A # A
    return ( cry_id - tva * tva_rms * 2.) / cry_id # percentage

def calc_channeling_efficiency(beam_momentum_GeV,
                               beam_incoming_angle_mrad = 0.,
                               beam_particle_mass_GeV = 0.9382720813,
                               beam_charge = +1.,
                               crystal_length_m = 1.E-3,
                               crystal_bending_radius_m = 1.E10,
                               crystal_el_field_max_Gev_on_cm = 5.7,
                               crystal_pot_depth_eV = 16.,
                               crystal_interplanar_distance_A = 1.92,
                               crystal_Z = 14.,
                               crystal_ionization_energy_eV = 172.,
                               thermal_vibration_amplitude_A  = 0.075,
                               thermal_vibration_rms_number   = 2.5):
    '''
    Description:
    
    Reference:
        Crystal Channeling and Its Application at High-Energy Accelerators
        Authors: Biryukov, Valery M., Chesnokov, Yuri A., Kotov, Vladilen I.
        https://www.springer.com/it/book/9783540607694

        Steering efficiency of a ultrarelativistic proton beam in a thin bent
        crystal
        Authors: E. Bagli et al.
        https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-014-2740-7

    '''

    beam_p      = beam_momentum_GeV
    beam_m      = beam_particle_mass_GeV
    
    beam_energy = get_energy(beam_p,beam_m)
    beam_gamma  = get_gamma(beam_energy,beam_m)
    beam_beta   = get_beta(beam_gamma)

    ia  = beam_incoming_angle_mrad
    pb  = beam_momentum_GeV*beam_beta
    cbr = crystal_bending_radius_m
    
    ga  = get_geometrical_acceptance(crystal_interplanar_distance_A,
                                     thermal_vibration_amplitude_A,
                                     thermal_vibration_rms_number)
    
    dl_e  = calc_dechanneling_length_e_m(beam_momentum_GeV,
                                         beam_particle_mass_GeV,
                                         crystal_bending_radius_m,
                                         crystal_el_field_max_Gev_on_cm,
                                         crystal_interplanar_distance_A,
                                         crystal_Z,
                                         crystal_ionization_energy_eV)
    
    cr  = get_critical_radius_m(pb,
                                crystal_el_field_max_Gev_on_cm,)

    dl_n = calc_dechanneling_length_n_m(pb)
    
    ca  = get_critical_angle_mrad(pb,
                                  crystal_bending_radius_m,
                                  crystal_pot_depth_eV,
                                  crystal_el_field_max_Gev_on_cm)
    eff_e = np.exp(- crystal_length_m / dl_e) 
    eff_n = np.exp(- crystal_length_m / dl_n)
    
    eff  = np.sqrt(np.abs(1.  - np.square(ia/ca))) * (1. - cr/cbr)

    if beam_charge < 0.:
        eff *= eff_n
    else:
        eff *= (ga * eff_e + (1. - ga) * eff_n)
    
    if hasattr(eff, "__len__"):
        eff[np.where( np.abs(ia) > ca )] = 0
    else:
        if np.abs(ia) > ca :
            return 0.
            
    if hasattr(cbr, "__len__") or hasattr(cr, "__len__"):
        eff[np.where( np.abs(cbr) < cr )] = 0
    else:
        if np.abs(cbr) < cr:
            return 0.
            
    return eff

def get_channeling_parameter_table(
            beam_momentum_GeV,
            beam_incoming_angle_mrad = 0.,
            beam_particle_mass_GeV = 0.9382720813,
            beam_charge = +1.,
            crystal_length_m = 1.E-3,
            crystal_bending_radius_m = 1.E10,
            crystal_el_field_max_Gev_on_cm = 5.7,
            crystal_pot_depth_eV = 16.,
            crystal_interplanar_distance_A = 1.92,
            crystal_Z = 14.,
            crystal_ionization_energy_eV = 172.,
            thermal_vibration_amplitude_A  = 0.075,
            thermal_vibration_rms_number   = 2.5):     
    
    beam_p      = beam_momentum_GeV
    beam_m      = beam_particle_mass_GeV
    
    beam_energy = get_energy(beam_p,beam_m)
    beam_gamma  = get_gamma(beam_energy,beam_m)
    beam_beta   = get_beta(beam_gamma)

    pb  = beam_momentum_GeV*beam_beta

    ch_ang_urad = get_deflection_angle_rad(
          crystal_length_m,
          crystal_bending_radius_m)*1.E6
    ch_crt_urad = get_critical_angle_mrad(
          momentum_velocity_GeV = pb,
          crystal_pot_depth_eV  = crystal_pot_depth_eV)*1.E3
    
    ch_eff = calc_channeling_efficiency(
            beam_momentum_GeV,
            beam_incoming_angle_mrad,
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
        
    dl_e_mm = calc_dechanneling_length_e_m(
        beam_momentum_GeV = beam_momentum_GeV,
        beam_particle_mass_GeV = beam_particle_mass_GeV,
        crystal_bending_radius_m = crystal_bending_radius_m,
        crystal_el_field_max_Gev_on_cm = crystal_el_field_max_Gev_on_cm,
        crystal_interplanar_distance_A = crystal_interplanar_distance_A,
        crystal_Z = crystal_Z,
        crystal_ionization_energy_eV = crystal_ionization_energy_eV
                                                   )*1.E3
    dl_n_mm = calc_dechanneling_length_n_m(
        momentum_velocity_GeV = pb)*1.E3
  
    da_e_urad = dl_e_mm / crystal_bending_radius_m * 1.E3
    da_n_urad = dl_n_mm / crystal_bending_radius_m * 1.E3
    
    print('Length [mm] ' + '%2.2f' % (crystal_length_m*1.E3))
    print('Radius [m] ' + '%2.2f' % (crystal_bending_radius_m))
    print(r'Defl. Ang. [µrad] ' + '%2.2f' % (ch_ang_urad))
    print(r'Max. Defl. Eff. [%] ' + '%2.2f' % (ch_eff))
    print('Crit. Ang. [µrad] ' + '%2.2f' % (ch_crt_urad))
    print('El. Dech. Length [mm] ' + '%2.2f' % (dl_e_mm))
    print('Nucl. Dech. Length [mm] ' + '%2.2f' % (dl_n_mm))
    print('El. Dech. Length [µrad] ' + '%2.2f' % (da_e_urad))
    print('Nucl. Dech. Length[µrad] ' + ' %2.2f' % (da_n_urad))
