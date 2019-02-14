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
                
    The collected experimental data contains:
        xN = position X at the N detector
        yN = position Y at the N detector
        
        
@author: enricobagli
"""

import numpy as np
from os import listdir
from os.path import isfile, join
import re
import pandas as pd
import uproot

def load_file_dat(runnumber,
                  file_path,
                  column_names):
    """ Load data files 
    
    :param runnumber: run number
    :param file_path: file path
    :param column_names: variable names
    
    :type runnumber: int
    :type file_path: string
    :type column_names: list of strings
    
    :return: pandas DataFrame with the data files
    """
    list_ = []
    for f in listdir(file_path):
        if isfile(join(file_path,f)) and f.endswith('dat'):
            if re.match(r'.*'+str(runnumber)+'.*\.dat', f):
                df = pd.read_csv(join(file_path,f),
                                 delim_whitespace=True,
                                 names = column_names,
                                 header=0)
                list_.append(df)
    return pd.concat(list_)

from scipy.optimize import minimize

def get_new_coord(hit,
                  hit_ortho,
                  position_shift,
                  angle_rotation):
    """ Get coordinates in a different reference system
    
    Small angle approximation is used in the calculation. The rotation is 
    performed before the translation of the coordinate system. The numpy array
    must have the same dimensions.
    
    :param hit: hit position
    :param hit_ortho: orthogonal hit position
    :param position_shift: shift of the hit position
    :param angle_rotation: rotation of the hit position

    :type hit: numpy array of float
    :type hit_ortho: numpy array of float
    :type position_shift: numpy array of float
    :type angle_rotation: float
    
    :return: numpy array of hit position in a rotated and translated ref system

    """
    return hit - position_shift - angle_rotation*hit_ortho

def get_projection(hit_ref_1,
                   hit_ref_2,
                   distance_ref,
                   distance):
    """ Get track projection point
    
    :param hit_ref_1: hit position on the first reference detector
    :param hit_ref_2: hit position on the second reference detector
    :param distance_ref: distance between the two reference detectors
    :param distance: distance of the projection point from first detector

    :type hit_ref_1: numpy array of float
    :type hit_ref_2: numpy array of float
    :type distance_ref: float
    :type distance: float
    
    :return: numpy array of the projected points
    """
    return (hit_ref_1 + (hit_ref_2 - hit_ref_1) / distance_ref * distance)

def reconstruct_vertex(x1,x2,x3,x4,
                       y1,y2,y3,y4):
    """ Reconstruct the vertex of the 12 and 34 tracks
    
    The vertex is reconstructed between the second and the third detector.
    The experimental setup is supposed to have a set of two detector for the
    incoming particle detection and  a set of two detectors for the outgoing
    particle detection.
    
    :param x1: horizontal position on the first detector
    :param x2: horizontal position on the second detector
    :param x3: horizontal position on the third detector
    :param x4: horizontal position on the fourth detector
    :param y1: vertical position on the first detector
    :param y2: vertical position on the second detector
    :param y3: vertical position on the third detector
    :param y4: vertical position on the fourth detector

    :type x1: numpy array of float
    :type x2: numpy array of float
    :type x3: numpy array of float
    :type x4: numpy array of float
    :type y1: numpy array of float
    :type y2: numpy array of float
    :type y3: numpy array of float
    :type y4: numpy array of float

    :return: numpy array of the reconstructed vertex
    """    
    x_part1 = (x1*x3*(y2 - y4) + x1*x4*(y3 - y2) + x2*x3*(y4 - y1) + x2*x4*(y1 - y3))
    x_part2 = ((x1 - x2)*(y3 - y4) + x3*(y2 - y1) + x4*(y1 - y2)) 
    x = x_part1 / x_part2
    y_part1 = (x1*y2*y3 - x1*y2*y4 + x2*y1*(y4 - y3) - x3*y1*y4 + x3*y2*y4 + x4*y3*(y1 - y2))
    y_part2 = ((x1 - x2)*(y3 - y4) + x3*(y2 - y1) + x4*(y1 - y2))
    y = y_part1 / y_part2

    return np.array(x) , np.array(y)

def compute_residuals(data,
                      distances,
                      position_shift,
                      angle_rotation):
    """ Compute the residuals on the tracks
    
    :param data: hit positions
    :param distances: distances between the detectors
    :param position_shift: shift of the hit positions
    :param angle_rotation: rotation of the hit positions

    :type data: pandas DataFrame
    :type distances: numpy array of float
    :type position_shift: numpy array of float
    :type angle_rotation: numpy array of float
    
    :return: two numpy arrays of horizontal and vertical residuals
    """    
    
    n_planes = len(distances)

    hit_ref_x = np.zeros([n_planes,data.x1.shape[0]])
    hit_ref_y = np.zeros([n_planes,data.x1.shape[0]])
   
    for i in range(n_planes):
        hit_ref_x[i] = get_new_coord(
                hit            = data['x'+str(i+1)],
                hit_ortho      = data['y'+str(i+1)], 
                position_shift = position_shift[i],
                angle_rotation = angle_rotation[i]
                )
        

        hit_ref_y[i] = get_new_coord(
                hit            = data['y'+str(i+1)],
                hit_ortho      = data['x'+str(i+1)], 
                position_shift = position_shift[i+4],
                angle_rotation = -angle_rotation[i]
                )
        
    expected_x = np.zeros([n_planes,data.x1.shape[0]])
    expected_y = np.zeros([n_planes,data.x1.shape[0]])
    
    for i in range(n_planes):
        expected_x[i] = get_projection(
                hit_ref_1    = hit_ref_x[0],
                hit_ref_2    = hit_ref_x[n_planes-1],
                distance_ref = distances[n_planes-1] - distances[0],
                distance     = distances[i] - distances[0])
        
        expected_y[i] = get_projection(
                hit_ref_1    = hit_ref_y[0],
                hit_ref_2    = hit_ref_y[n_planes-1],
                distance_ref = distances[n_planes-1] - distances[0],
                distance     = distances[i] - distances[0])

    residuals_x  = np.zeros([n_planes,data.x1.shape[0]])
    residuals_y  = np.zeros([n_planes,data.x1.shape[0]])
    for i in range(n_planes):
        residuals_x[i] = (hit_ref_x[i] - expected_x[i])
        residuals_y[i] = (hit_ref_y[i] - expected_y[i])
    return residuals_x, residuals_y


def compute_residual_sum(x,
                         *args):
    """ Compute the sum of the residuals on the tracks
    
    :param x: shift and rotations
    :param *args: hit positions and distances of the detectors
    
    :type x: numpy array of floats
    :type *args: list
            [0] pandas DataFrame
            [1] numpy array of float

    :return: float with the sum of the residuals
    """    
    
    data     = args[0]
    distance = args[1]
    position_shift = x[0 : 8]
    angle_rotation = x[8 :12]
    
    residuals_x, residuals_y = compute_residuals(
            data,
            distance,
            position_shift,
            angle_rotation
            )
    
    return (np.sum(np.fabs(residuals_x)) +  np.sum(np.fabs(residuals_y)))
   

def compute_alignment_parameter(data,
                                distances,
                                min_function  = compute_residual_sum):
    """ Compute the alignment parameters of the detectors
    
    :param data: hit positions
    :param distances: distances of the detectors
    :param min_function: function to be minimized
    
    :type data: pandas DataFrame
    :type distances: numpy array of float
    :type min_function: function with x and *args
        
    :return: numpy array of float of the alignment parameters
    """    

    args = (data,
            distances)
    
    bounds = (
        (-150,+150),
        (0,0),
        (-150,+150),
        (-150,+150),

        (-150,+150),
        (0,0),
        (-150,+150),
        (-150,+150),

        (-0.5,0.5),
        (0.,0),
        (-0.5,0.5),
        (-0.5,0.5)
    )
    
    x0 = [np.mean(data.x1 - data.x2),
          0.,
          np.mean(data.x3 - data.x2),
          np.mean(data.x4 - data.x2),
          np.mean(data.y1 - data.y2),
          0.,
          np.mean(data.y3 - data.y2),
          np.mean(data.y4 - data.y2),
          0.,
          0.,
          0.,
          0.]
    
    transformations = minimize(min_function,
                               x0 = x0,
                               args   = args,
                               bounds = bounds,
                               tol=1e-6) 
    return transformations

def transform_data(data,
                   transformations):
    """ Transform detector data
    
    :param data: hit positions
    :param transformations: alignment parameters
    
    :type data: pandas DataFrame
    :type transformations: numpy array of float
        
    :return: pandas DataFrame with the transformed coordinates
    """    
    
    n_planes = int(len(transformations)/3.)
   
    for i in range(n_planes):
        data['x'+str(i+1)+'n'] = get_new_coord(
                data['x'+str(i+1)],
                data['y'+str(i+1)],
                transformations[i],
                transformations[n_planes*2+i])
                                
        data['y'+str(i+1)+'n'] = get_new_coord(
                data['y'+str(i+1)],
                data['x'+str(i+1)],
                transformations[n_planes+i],
                -transformations[n_planes*2+i])
        
    if 'x3s' in data.columns:
        for i in range(2,4):
            data['x'+str(i+1)+'sn'] = get_new_coord(
                    data['x'+str(i+1)+'s'],
                    data['y'+str(i+1)+'s'],
                    transformations[i],
                    transformations[n_planes*2+i])
        
            data['y'+str(i+1)+'sn'] = get_new_coord(
                    data['y'+str(i+1)+'s'],
                    data['x'+str(i+1)+'s'],
                    transformations[n_planes+i],
                    -transformations[n_planes*2+i])
    return data


def compute_ang(data,
                distance_12,
                distance_34):
    """ Compute the incoming and outgoing angle of the particles
    
    :param data: hit positions
    :param distance_12: distance between the detectors of the first leg
    :param distance_34: distance between the detectors of the second leg
    
    :type data: pandas DataFrame
    :type distance_12: float
    :type distance_34: float
        
    :return: pandas DataFrame with the incoming and outgoing angles
    """    

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

    if 'x3s' in data.columns:
        data['ang_out_s_x_urad'] = (data['x4sn']-data['x3sn'])*1.E4 / distance_34
        data['ang_out_s_x_urad'] = data['ang_out_s_x_urad'] - shiftx
        data['ang_out_s_y_urad'] = (data['y4sn']-data['y3sn'])*1.E4 / distance_34
        data['ang_out_s_y_urad'] = data['ang_out_s_y_urad'] - shifty
        data['defl_s_x_urad']  = data['ang_in_x_urad'] - data['ang_out_s_x_urad']
        data['defl_s_y_urad']  = data['ang_in_y_urad'] - data['ang_out_s_y_urad']

    data['pos_in_y_mm']  = data['y1n']*1.E1 
    data['pos_in_y_mm'] += data['ang_in_y_urad'] * distance_12 * 1.E-3
    return data
    

def read_ROOT_file(filename):
    """ Read ROOT file created with Imperial College experimental setup
    
    :param filename: name of the input file
    
    :type filename: string
        
    :return: pandas DataFrame
    """    
    file = uproot.open(filename)
    tree = file["simpleEvent"]
    Tracks = tree.arrays(['Tracks'])
    SingleTrack = tree.arrays(['SingleTrack'])
    df = pd.DataFrame.from_dict(Tracks[b'Tracks'])
    return df[SingleTrack[b'SingleTrack']==1].copy()

def transform_ROOT_file(df):
    """ Rename columns of the DataFrame created from ROOT files.
    
    :param df: detector data
    
    :type df: pandas DataFrame
        
    :return: pandas DataFrame
    """
    
    df['ang_in_x_urad'] = df['thetaIn_x']*1.E6
    df['ang_in_y_urad'] = df['thetaIn_y']*1.E6
    df['ang_out_x_urad'] = df['thetaOut_x']*1.E6
    df['ang_out_y_urad'] = df['thetaOut_y']*1.E6
    df['defl_x_urad'] = df.ang_in_x_urad - df.ang_out_x_urad
    df['defl_y_urad'] = df.ang_in_y_urad - df.ang_out_y_urad
    df['pos_in_x_mm'] = df['d0_x']
    df['pos_in_y_mm'] = df['d0_y']
    return df[df.columns[-8:]].copy()
    
