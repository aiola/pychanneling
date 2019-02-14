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
    column_names = ['eventID',
                    'trackID',
                    'layerID',
                    'pos_x_um',
                    'pos_y_um',
                    'pos_z_um',
                    'spin_x',
                    'spin_y',
                    'spin_z',
                    'kin_e_MeV',
                    'pdg',
                    'charge']
    dtypes = {}
    
    for col in column_names:
        dtypes[col] = np.float32
        
    df = pd.read_csv(csv_geant4_file,
                     names=column_names,
                     sep=',',
                     comment='#',
                     dtype=dtypes,
                     index_col=None)
    
    return df

def read_file_compressed(csv_geant4_file):
    '''
    Description:
        Read compressed Geant4 simulation output file.
        
    '''
    column_names = ['eventID',
                    'trackID',
                    'layerID',
                    'pos_x_um',
                    'pos_y_um',
                    'pos_z_um',
                    'spin_x',
                    'spin_y',
                    'spin_z',
                    'kin_e_MeV',
                    'pdg',
                    'charge']
    dtypes = {}
    
    for col in column_names:
        dtypes[col] = np.float32
        
    df = pd.read_csv(csv_geant4_file,
                     compression='gzip',
                     names=column_names,
                     sep=',',
                     header=0,
                     comment='#',
                     dtype=dtypes,
                     index_col=None,
                     error_bad_lines=False)
    
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
     
    
def write_crystal_mac(
    file,
    crystal_length_cm,
    crystal_bending_radius_m,
    crystal_width_cm  = 10., 
    crystal_height_cm = 10.,
    crystal_material = 'G4_Si',
    crystal_datafile_name = 'data/Si220pl',
    crystal_rotation_x = 0.,
    crystal_rotation_y = 0.,
    crystal_rotation_z = 0.
):
    try:
        string = '/xtal/setMaterial ' + crystal_material + '\n'
        file.write(string)

        string = '/xtal/setSize '
        string += str(crystal_width_cm) + ' '
        string += str(crystal_height_cm) + ' '
        string += str(crystal_length_cm) + ' '
        string += 'cm\n'
        file.write(string)

        string = '/xtal/setAngle '
        string += str(crystal_rotation_x*1.E-6) + ' '
        string += str(crystal_rotation_y*1.E-6) + ' '
        string += str(crystal_rotation_z*1.E-6) + ' '
        string += 'rad\n'
        file.write(string)

        string = '/xtal/setBR '+ str(crystal_bending_radius_m) + ' 0.0 0.0 m\n'
        file.write(string)

        string = '/xtal/setEC data/' + crystal_datafile_name + '\n'
        file.write(string)
        return True
    except:
        return False
    
    return False

def write_det_mac(
    file,
    detector_material   = 'G4_Galactic',
    detector_length_mm   = 0.640,
    detector_width_mm   = 93.,
    detector_1_pos_z_cm = -20.,
    detector_2_pos_z_cm = -10.,
    detector_3_pos_z_cm = +10.,
    detector_4_pos_z_cm = +20.,
    detector_5_pos_z_cm = +30.
):
    try:
        string = '/mydet/setDetMaterial ' + detector_material +'\n'
        file.write(string)

        string = '/mydet/setSize '
        string += str(detector_width_mm) + ' '
        string += str(detector_width_mm) + ' '
        string += str(detector_length_mm) + ' '
        string += ' mm\n'
        file.write(string)

        string = '/mydet/setDistance1 '
        string += str(detector_1_pos_z_cm) + ' '
        string += ' cm\n'
        file.write(string)

        string = '/mydet/setDistance2 '
        string += str(detector_2_pos_z_cm) + ' '
        string += ' cm\n'
        file.write(string)

        string = '/mydet/setDistance3 '
        string += str(detector_3_pos_z_cm) + ' '
        string += ' cm\n'
        file.write(string)

        string = '/mydet/setDistance4 '
        string += str(detector_4_pos_z_cm) + ' '
        string += ' cm\n'
        file.write(string)

        string = '/mydet/setDistance5 '
        string += str(detector_5_pos_z_cm) + ' '
        string += ' cm\n'
        file.write(string)
        return True
    except:
        return False
    
def write_beam_mac(
    file,
    beam_energy_GeV = 400.,
    beam_particle_type = 'proton',
    beam_polarization_x = 0.,
    beam_divergence_x_urad = 0.,
    beam_divergence_y_urad = 0.,
    beam_position_z_cm = -21.
):
    try:
        string = '/gps/particle ' + beam_particle_type + '\n'
        file.write(string)

        string = '/gps/ene/mono '+ str(beam_energy_GeV) + ' GeV\n'
        file.write(string)

        string = '/gps/polarization '+ str(beam_polarization_x) + ' 0. 0.\n'
        file.write(string)

        file.write('/gps/ang/rot1 1 0 0\n')
        file.write('/gps/ang/rot2 0 -1 0\n')
        file.write('/gps/ang/type beam2d\n')

        string = '/gps/ang/sigma_x '+ str(beam_divergence_x_urad)+'E-6 rad\n'
        file.write(string)

        string = '/gps/ang/sigma_y '+ str(beam_divergence_y_urad)+'E-6 rad\n'
        file.write(string)

        file.write('/gps/pos/type Point\n')

        string = '/gps/pos/centre 0. 0. ' + str(beam_position_z_cm) + ' cm\n'
        file.write(string)
        return True
    except:
        return False


def write_mac(
    filename_mac,
    filename_csv,
    crystal_length_cm,
    crystal_bending_radius_m,
    beam_number_of_particles,
    crystal_width_cm  = 10., 
    crystal_height_cm = 10.,
    crystal_material = 'G4_Si',
    crystal_datafile_name = 'data/Si220pl',
    crystal_rotation_x = 0.,
    crystal_rotation_y = 0.,
    crystal_rotation_z = 0.,
    beam_energy_GeV = 400.,
    beam_particle_type = 'proton',
    beam_polarization_x = 0.,
    beam_divergence_x_urad = 0.,
    beam_divergence_y_urad = 0.,
    beam_position_z_cm = -21.,
    detector_material   = 'G4_Galactic',
    detector_length_mm   = 0.640,
    detector_width_mm   = 93.,
    detector_1_pos_z_cm = -20.,
    detector_2_pos_z_cm = -10.,
    detector_3_pos_z_cm = +10.,
    detector_4_pos_z_cm = +20.,
    detector_5_pos_z_cm = +30.
):
    try:
        file = open(filename_mac,"w")

        debug_crystal = write_crystal_mac(
            file,
            crystal_length_cm,
            crystal_bending_radius_m,
            crystal_width_cm, 
            crystal_height_cm,
            crystal_material,
            crystal_datafile_name,
            crystal_rotation_x,
            crystal_rotation_y,
            crystal_rotation_z)
        
        debug_det = write_det_mac(
            file,
            detector_material,
            detector_length_mm,
            detector_width_mm,
            detector_1_pos_z_cm,
            detector_2_pos_z_cm,
            detector_3_pos_z_cm,
            detector_4_pos_z_cm,
            detector_5_pos_z_cm)
        
        file.write('/run/initialize\n')
        file.write('/filename/set ' + filename_csv + '\n')

        debug_beam = write_beam_mac(
            file,
            beam_energy_GeV,
            beam_particle_type,
            beam_polarization_x,
            beam_divergence_x_urad,
            beam_divergence_y_urad,
            beam_position_z_cm)
        
        file.write('/run/printProgress 100\n')
        file.write('/run/beamOn ' + str(beam_number_of_particles) + '\n')
        file.close()
        return True
    except:
        return False
    
    
    
def reconstruct_angles(df):
    '''
    '''
    dfg = df.groupby(['trackID','eventID']).first()
    dfg['eventID'] = dfg.index.get_level_values(1)
    dfg['trackID'] = dfg.index.get_level_values(0)    
    
    df_hit_0 = df[['eventID','trackID','pos_x_um','pos_y_um','pos_z_um','kin_e_MeV']][df['layerID'] == 0]
    df_hit_1 = df[['eventID','trackID','pos_x_um','pos_y_um','pos_z_um','kin_e_MeV']][df['layerID'] == 1]
    df_hit_2 = df[['eventID','trackID','pos_x_um','pos_y_um','pos_z_um','kin_e_MeV']][df['layerID'] == 2]
    df_hit_3 = df[['eventID','trackID','pos_x_um','pos_y_um','pos_z_um','kin_e_MeV']][df['layerID'] == 3]

    df_hit_01 = df_hit_0.merge(df_hit_1, on=['trackID','eventID'],suffixes = ['_0','_1'],how='outer')
    df_hit_23 = df_hit_2.merge(df_hit_3, on=['trackID','eventID'],suffixes = ['_2','_3'],how='outer')
    df_hit    = df_hit_01.merge(df_hit_23, on=['trackID','eventID'],how='outer')
    df_hit    = df_hit.merge(dfg[['eventID','trackID','pdg','charge']],on=['eventID','trackID'])
    
    df_hit['ang_in_x_urad'] = (df_hit['pos_x_um_1'] - df_hit['pos_x_um_0']) / (df_hit['pos_z_um_1'] - df_hit['pos_z_um_0']) * 1.E6
    df_hit['ang_in_y_urad'] = (df_hit['pos_y_um_1'] - df_hit['pos_y_um_0']) / (df_hit['pos_z_um_1'] - df_hit['pos_z_um_0']) * 1.E6
    
    df_hit['ang_out_x_urad'] = (df_hit['pos_x_um_3'] - df_hit['pos_x_um_2']) / (df_hit['pos_z_um_3'] - df_hit['pos_z_um_2']) * 1.E6
    df_hit['ang_out_y_urad'] = (df_hit['pos_y_um_3'] - df_hit['pos_y_um_2']) / (df_hit['pos_z_um_3'] - df_hit['pos_z_um_2']) * 1.E6
    
    df_hit['defl_x_urad'] = df_hit['ang_in_x_urad'] - df_hit['ang_out_x_urad']
    df_hit['defl_y_urad'] = df_hit['ang_in_y_urad'] - df_hit['ang_out_y_urad']
    
    df_hit = df_hit.drop(
        list(df_hit.filter(like='pos').columns.values),
        axis=1)

    return df_hit

def mask_primary(df):
    mask_track_primary = (df['trackID']==1)
    return mask_track_primary

def mask_alive(df):
    mask_energy_in = (~df['kin_e_MeV_0'].isnull()) & (~df['kin_e_MeV_1'].isnull())
    mask_energy_out = (~df['kin_e_MeV_2'].isnull()) & (~df['kin_e_MeV_3'].isnull())
    return mask_energy_in & mask_energy_out

def mask_primary_alive(df):
    return mask_primary(df) & mask_alive(df)

def mask_primary_dead(df):
    mask_energy_in = (~df['kin_e_MeV_0'].isnull()) & (~df['kin_e_MeV_1'].isnull())
    mask_not_energy_out = (df['kin_e_MeV_2'].isnull()) & (df['kin_e_MeV_3'].isnull())
    return mask_primary(df) & mask_not_energy_out & mask_energy_in & ~mask_alive(df)

def mask_secondary_after_target(df):
    mask_not_energy_in = (df['kin_e_MeV_0'].isnull()) & (df['kin_e_MeV_1'].isnull())
    mask_energy_out = (~df['kin_e_MeV_2'].isnull()) & (~df['kin_e_MeV_3'].isnull())
    return mask_not_energy_in & mask_energy_out & ~mask_primary(df)

def add_angin_to_secondaries(
    df_total,
    df_secondaries):
    tIDu = df_total.trackID.unique()
    mask = df_total.trackID.isin(tIDu) & mask_primary(df_total)
    df_angin = df_total[['eventID','ang_in_x_urad','ang_in_y_urad']][mask]
    df_sec = df_secondaries.drop(['ang_in_x_urad','ang_in_y_urad'],axis=1)
    df_sec = df_sec.merge(df_angin,on='eventID',how='left')
    df_sec['defl_x_urad'] = df_sec['ang_in_x_urad'] - df_sec['ang_out_x_urad']
    df_sec['defl_y_urad'] = df_sec['ang_in_y_urad'] - df_sec['ang_out_y_urad']
    return df_sec

