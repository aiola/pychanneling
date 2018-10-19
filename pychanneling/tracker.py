#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Tracker functions
    
    The reference system is the following:
        Z = beam direction
        X = direction parallel to the floor and perpendicular to Z
        Y = direction perpendicular to the floor and to Z
    
    The pandas DataFrame for the description of the particle has the following
    columns for the analysis:
        ang_in_x_urad = incoming X angle [urad]
        ang_in_y_urad = incoming Y angle [urad]
                
        ang_out_x_urad = outgoing X angle [urad]
        ang_out_y_urad = outgoing Y angle [urad]

        defl_x_urad = X deflection angle after the crystal [urad]
        defl_y_urad = Y deflection angle after the crystal [urad]
                
        pos_in_x_mm = position X at the crystal entrance [um]
        pos_in_y_mm = position Y at the crystal entrance [um]
                
    The experimental data collected with the Insubria tracker may also contain
    other information:
        xN = position X at the N detector
        yN = position Y at the N detector
        
        clusterxN = hit cluster on the N detector for X
        clusteryN = hit cluster on the N detector for Y
        
@author: enricobagli
"""

import numpy as np
from os import listdir
from os.path import isfile, join
import re
import pandas as pd
import uproot

def get_SELDOM_column_names():
    names = ['x1','y1','x2','y2','x3','y3','x4','y4']
    [names.append('cluster'+str(x)) for x in range(8)]
    [names.append('mov'+str(x)) for x in range(3)]
    [names.append('info'+str(x)) for x in range(3)]
    return names

def load_file_dat(runnumber,
                  mypath,
                  column_names):
    list_ = []
    for f in listdir(mypath):
        if isfile(join(mypath,f)) and f.endswith('dat'):
            if re.match(r'.*'+str(runnumber)+'.*\.dat', f):
                df = pd.read_csv(join(mypath,f),
                                 delim_whitespace=True,
                                 names = column_names,
                                 header=0)
                list_.append(df)
    return pd.concat(list_)

def get_clean_hit_data(data,
                       column_names):
    df_c = data.copy()
    for col in column_names:
        df_c = df_c[(df_c[col]>-1000.)&(df_c[col]<1000.)]
    return df_c

def filter_single_cluster_data(data,
                                    max_cluster = 2):
    df_c = data.copy()
    for col in df_c.columns:
        if col[:7] == 'cluster':
            df_c = df_c[df_c[col]<=max_cluster]
    return df_c

from scipy.optimize import minimize

def get_new_coord(hit,
                  hit_ortho,
                  position_shift,
                  angle_rotation):
    return hit - position_shift - angle_rotation*hit_ortho

def get_projection(hit_ref_1,
                   hit_ref_2,
                   distance_ref,
                   distance):
    return (hit_ref_1 + (hit_ref_2 - hit_ref_1) / distance_ref * distance)

def norm_L1(x1,x2):
    return np.abs(x1-x2)

def norm_L2(x1,x2):
    return np.square(x1-x2)

def get_residual_sum(x,
                     *args):
    
    data     = args[0]
    distance = [args[1],
                args[2],
                args[3]]
    cost_function  = args[4]
    position_shift = x[0 : 8]
    angle_rotation = x[8 :12]
    
    hit_ref        = np.zeros([8,data.x1.shape[0]])
    
    for i in range(4):
        hit_ref[i]   = get_new_coord(hit            = data['x'+str(i+1)],
                                     hit_ortho      = data['y'+str(i+1)], 
                                     position_shift = position_shift[i],
                                     angle_rotation = angle_rotation[i])

        hit_ref[i+4] = get_new_coord(hit            = data['y'+str(i+1)],
                                     hit_ortho      = data['x'+str(i+1)], 
                                     position_shift = position_shift[i+4],
                                     angle_rotation = -angle_rotation[i])
    expected = np.zeros([4,data.x1.shape[0]])
    
    residuals = np.zeros([4,data.x1.shape[0]])
    
    for i in range(2):
        expected[i]   = get_projection(hit_ref_1    = hit_ref[0],
                                       hit_ref_2    = hit_ref[3],
                                       distance_ref = distance[2],
                                       distance     = distance[i])
        expected[i+2] = get_projection(hit_ref_1    = hit_ref[4],
                                       hit_ref_2    = hit_ref[7],
                                       distance_ref = distance[2],
                                       distance     = distance[i])
        residuals[i]   = cost_function(hit_ref[i+1]   , expected[i])
        residuals[i+2] = cost_function(hit_ref[i+1+4] , expected[i+2])
    return np.sum(residuals)

def get_transormation_parameter(data,
                                distance_12,
                                distance_13,
                                distance_14,
                                cost_function = norm_L1):
    args = (data,
            distance_12,
            distance_13,
            distance_14,
            cost_function)
    
    bounds = (
        (-15,+15),
        (-15,+15),
        (-15,+15),
        (-15,+15),

        (-15,+15),
        (-15,+15),
        (-15,+15),
        (-15,+15),

        (-0.5,0.5),
        (-0.5,0.5),
        (-0.5,0.5),
        (-0.5,0.5)
    )
    transformations = minimize(get_residual_sum,
                               x0 = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                               args   = args,
                               bounds = bounds,
                               tol=1e-6) 
    return transformations


def transform_data(data,
                   transformations,
                   distance_12,
                   distance_34):
    
    for i in range(4):
        data['x'+str(i+1)+'n'] = get_new_coord(data['x'+str(i+1)],
                                               data['y'+str(i+1)],
                                               transformations[i],
                                               transformations[8+i])
                                
        data['y'+str(i+1)+'n'] = get_new_coord(data['y'+str(i+1)],
                                               data['x'+str(i+1)],
                                               transformations[4+i],
                                               -transformations[8+i])
                                    
    data['ang_in_x_urad']  = (data['x2n']-data['x1n'])*1.E4 / distance_12
    data['ang_out_x_urad'] = (data['x4n']-data['x3n'])*1.E4 / distance_34

    shiftx = np.mean(data['ang_in_x_urad'])
    data['ang_out_x_urad'] = data['ang_out_x_urad'] - shiftx
    data['ang_in_x_urad']  = data['ang_in_x_urad']  - shiftx
    
    data['ang_in_y_urad']  = (data['y2n']-data['y1n'])*1.E4 / distance_12
    data['ang_out_y_urad'] = (data['y4n']-data['y3n'])*1.E4 / distance_34

    shifty = np.mean(data['ang_in_y_urad'])
    data['ang_out_y_urad'] = data['ang_out_y_urad'] - shifty
    data['ang_in_y_urad']  = data['ang_in_y_urad']  - shifty

    data['defl_x_urad']  = data['ang_in_x_urad'] - data['ang_out_x_urad']
    data['defl_y_urad']  = data['ang_in_y_urad'] - data['ang_out_y_urad']
    
    data['pos_in_x_mm']  = data['x1n']*1.E1
    data['pos_in_x_mm'] += data['ang_in_x_urad'] * distance_12 * 1.E-3
    
    data['pos_in_y_mm']  = data['y1n']*1.E1 
    data['pos_in_y_mm'] += data['ang_in_y_urad'] * distance_12 * 1.E-3
    

def read_ROOT_file(filename):
    file = uproot.open(filename)
    tree = file["simpleEvent"]
    Tracks = tree.arrays(['Tracks'])
    SingleTrack = tree.arrays(['SingleTrack'])
    df = pd.DataFrame.from_dict(Tracks[b'Tracks'])
    return df[SingleTrack[b'SingleTrack']==1].copy()

def transform_ROOT_file(df):
    df['ang_in_x_urad'] = df['thetaIn_x']*1.E6
    df['ang_in_y_urad'] = df['thetaIn_y']*1.E6
    df['ang_out_x_urad'] = df['thetaOut_x']*1.E6
    df['ang_out_y_urad'] = df['thetaOut_y']*1.E6
    df['defl_x_urad'] = df.ang_in_x_urad - df.ang_out_x_urad
    df['defl_y_urad'] = df.ang_in_y_urad - df.ang_out_y_urad
    df['pos_in_x_mm'] = df['d0_x']
    df['pos_in_y_mm'] = df['d0_y']
    return df[df.columns[-8:]].copy()
    
''' OLD FUNCTIONS
def get_shift(series_1,
              series_2):
    return (series_1-series_2).mean()

def get_angle(series_1,
              series_2,
              distance):
    shift = get_shift(series_1,
                      series_2)
    angle = series_1 - (series_2 + shift)
    return angle

def process_detector_positions(det0_X,
                               det0_Y,
                               det1_X,
                               det1_Y,
                               det2_X,
                               det2_Y,
                               det01_distance,
                               det12_distance):

    angle_xin = get_angle(series_1 = det0_X,
                          series_2 = det1_X,
                          distance = det01_distance)

    angle_yin = get_angle(series_1 = det0_Y,
                          series_2 = det1_Y,
                          distance = det01_distance)

    angle_xot = get_angle(series_1 = det2_X,
                          series_2 = det1_X,
                          distance = det12_distance)

    angle_yot = get_angle(series_1 = det2_Y,
                          series_2 = det1_Y,
                          distance = det12_distance)

    return angle_xin, angle_yin, angle_xot, angle_yot
'''