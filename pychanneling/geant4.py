#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Support functions for pandas DataFrame created from Geant4 simulations.
    Simulation code can be found at:
        https://github.com/ebagli/G4_MC_CHANNELING
    
@author: Enrico Bagli
"""
import pandas as pd
import numpy as np

def read_file(csv_geant4_file):
    '''
    Description:
        Read Geant4 simulation output file.
        
    '''
    column_names = ['ang_in_x_urad',
                    'ang_in_y_urad',
                    'pos_in_x_mm',
                    'pos_in_y_mm',
                    'ang_out_x_urad',
                    'ang_out_y_urad',
                    'ef_x_eV_on_A',
                    'ef_y_eV_on_A',
                    'nud',
                    'eld',
                    'spin_x',
                    'spin_y',
                    'spin_z',
                    'kin_e_MeV']
                    
    df = pd.read_csv(csv_geant4_file,
                     names=column_names,
                     sep=',',
                     comment='#',
                     index_col=None)
    
    df['defl_x_urad']   = df.ang_in_x_urad - df.ang_out_x_urad
    df['defl_y_urad']   = df.ang_in_y_urad - df.ang_out_y_urad
    return df

        
def add_mdm_edm(df):
    '''
    Description:
        Compute MDM and EDM.
        
    Reference:
        Electromagnetic dipole moments of charged baryons with bent crystals
        at the LHC
        Authors: E. Bagli et al.
        https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-017-5400-x
    '''
    df['spin_x_art']  = df['spin_y']
    df['spin_y_art']  = df['spin_x']
    df['spin_z_art']  = df['spin_z']
    df['mdm']     = np.arctan2(df['spin_z_art'],df['spin_y_art'])/np.pi*180
    df['edm']     = np.arctan2(df['spin_x_art'],df['spin_y_art'])/np.pi*180
    
    
def add_pos_out(df,
                det_distance_mm = 1.):
    '''
    Description:
        Add outgoing position to the particles.
        The position depends from the distance.
    '''
    
    molt = 1.E-6*det_distance_mm
    df['pos_out_x_mm'] = df['pos_in_x_mm'] + df['ang_out_x_urad']*molt
    df['pos_out_y_mm'] = df['pos_in_y_mm'] + df['ang_out_y_urad']*molt


def add_torsion(df,
                torsion_urad_on_mm,
                pos_in_y_mm_rms = None):
    '''
    Description:
        Add torsion to the particle incoming angle.
        The modification of the incoming angle can be calculated as a function
        of the incoming position on the crystal or as a function of a Gaussian
        PDF with a user defined rms.
    '''
    
    if pos_in_y_mm_rms == None:
        pos_in_y_mm_T = df['pos_in_y_mm']
    else:
        pos_in_y_mm_T = np.random.normal(loc   = 0.,
                                         scale = pos_in_y_mm_rms,
                                         size  = df.shape[0])
    
    correction = pos_in_y_mm_T * torsion_urad_on_mm
    df['ang_in_x_torsion_urad'] = df['ang_in_x_urad'] + correction
     
    
