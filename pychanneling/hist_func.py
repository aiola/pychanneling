#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Operations on hist and bin_edges variables got from np.histogram function.

@author: enricobagli
"""

import numpy as np

def get_selection_mask(series,
                       selection_width,
                       selection_mean = 0.):
    '''
    Description:
        Get a mask for the selection of a portion of the Series
    '''
    try:
        if selection_width > 0.:
            mask = abs(series - selection_mean) < selection_width
            return mask

    except:
        print("Error: rms <= 0. OR feature does not exist\n")

def get_mean_x12(x1,x2):
    return (x1+x2)*0.5

def get_width_x12(x1,x2):
    return np.abs((x2-x1)*0.5)

def get_rotated(x,
                y,
                angle_deg = 0.):
    angle_rad = np.deg2rad(angle_deg)
    x_rot = x*np.cos(angle_rad) - y*np.sin(angle_rad)
    y_rot = x*np.sin(angle_rad) + y*np.cos(angle_rad)
    return x_rot, y_rot

def get_bin_size_array_from_edges(bin_edges):
    '''
    Description:
        Get bin size from the edges
    '''
    be = bin_edges
    return np.array([be[i+1]-be[i] for i in range(be.shape[0]-1)])

def get_bin_size_from_edges(bin_edges):
    '''
    Description:
        Get bin size from the edges
    '''
    return np.fabs(bin_edges[1]-bin_edges[0])

def get_bin_range(data,
                  values):
    
    if values[0] < data[0] or values == None:
        bin_xs_min = 0
    else:
        bin_xs_min = np.searchsorted(data, values[0])
    
    if values[1] > data[-1] or values == None:
        bin_xs_max = -1 
    else:
        bin_xs_max = np.searchsorted(data, values[1])

    return bin_xs_min, bin_xs_max

def get_centers_from_edges(bin_edges):
    '''
    Description:
        Get centers of the bins from their edges
    '''
    be = bin_edges
    return np.array([(be[i] + be[i+1])*0.5 for i in range(len(be)-1)])

def get_edges_from_centers(centers):
    '''
    Description:
        Get edges of the bins from their centers
    '''
    ce = centers
    bs = get_bin_size_from_edges(centers)
    be = [ce[i] - bs*0.5 for i in range(len(ce))]
    be.append(ce[-1] + bs*0.5)
    return np.array(be)

def reshape_vector(vector,
                   new_bins):
    '''
    Description:
        Reshape a vector
    '''
    return np.array(vector).reshape(len(vector)//new_bins,new_bins)
            
def rebin(hist,
          bin_edges,
          bins_new):
    '''
    Description:
        Rebin an hisogram.
    '''
    centers = get_centers_from_edges(bin_edges)
    
    hist_new    = np.array(reshape_vector(hist    , bins_new)).sum(-1)       
    centers_new = np.array(reshape_vector(centers , bins_new)).mean(-1) 
    
    bin_edges_new = get_edges_from_centers(centers_new)
    
    return hist_new, bin_edges_new
    
def normalize(hist,
              bin_edges,
              constant = 1):
    '''
    Description:
        Normalize the contents of an histogram.
    '''
    bin_size = get_bin_size_from_edges(bin_edges)
    norm = np.sum(hist) * bin_size
    hist_norm = np.array([x / norm for x in hist])
    return hist_norm * constant, bin_edges
    
def convolve_gaussian(hist,
                      bin_edges,
                      gaus_sigma = 2.):
    '''
    Description:
        Convolve the histogram distribution with a gaussian.
    '''
    from scipy.ndimage import gaussian_filter
    step = get_bin_size_from_edges(bin_edges)
    hist_new = gaussian_filter(hist, sigma=gaus_sigma/step)
    return hist_new, bin_edges

def save_hist_to_csv(filename,
                     hist,
                     bin_edges):
    centers = get_centers_from_edges(bin_edges)
    f = open(filename,'w')
    for c,h in zip(centers,hist):
        f.write(str(c)+','+str(h)+'\n')
    f.close()
    
def load_hist_from_csv(filename):
    hist = []
    centers = []
    f = open(filename,'r')
    content = f.readlines()
    for c in content:
        data = c.rstrip('\n').split(',')
        centers.append(float(data[0]))
        hist.append(float(data[1]))
    hist = np.array(hist)
    centers = np.array(centers)
    bin_edges = get_edges_from_centers(centers)
    return hist, bin_edges
   