# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:37:33 2024
sources.py 

This module provides functions related to the generation of optical sources. 
@author: Olga
"""

import numpy as np
from scipy.special import genlaguerre

def gaussian(wavelength = 600e-9, waist_radius = 100e-6, frame = 50e-6, n = 101):
    field = np.zeros((n, n), dtype = complex)
    
    windowX = 2 * (waist_radius + frame)
    xMin = -windowX / 2
    dx = windowX / (n - 1)
    
    for ix in range (0, n):
        x = xMin + ix * dx
        for iy in range (0, n):
            y = xMin + iy * dx
            field[ix, iy] = np.exp(-(x ** 2 + y ** 2) / (waist_radius ** 2))
    
    return field

def gaussianxy(wavelength = 600e-9, waist_radius_x = 100e-6, waist_radius_y = 100e-6, frame_x = 50e-6, frame_y = 50e-6, nx = 101, ny = 101):
    field = np.zeros((nx, ny), dtype = complex)
    
    windowX = 2 * (waist_radius_x + frame_x)
    windowY = 2 * (waist_radius_y + frame_y)
    xMin = -windowX / 2
    yMin = -windowY / 2
    dx = windowX / (nx - 1)
    dy = windowY / (ny - 1)
    
    for ix in range (0, nx):
        x = xMin + ix * dx
        for iy in range (0, ny):
            y = yMin + iy * dy
            field[ix, iy] = np.exp(-(x ** 2) / (waist_radius_x ** 2)) * np.exp(-(y ** 2) / (waist_radius_y ** 2))
            
    return field

def laguerre_gauss_paraxial_parameters(wavelength = 600e-9, waist_radius = 100e-6, radial_order = 0, azimuthal_order = 0, z = 0.0):
    """
    This function calculates the parameters of a Laguerre-Gauss beam using 
    the paraxial approximation. 
    
    Parameters: 
        wavelength (double): The wavelength of the radiation. 
        waist_radius (double): The waist radius w_0 of the Laguerre-Gauss beam. 
        radial_order (int): The radial order of the Laguerre-Gauss. Can take zero and positive integer values. 
        azimuthal_order (int): The azimuthal order of the Laguerre-Gauss. Can take zero, as well as both negative and positive integer values. 
        z (double): The z (longitudinal) coordinate where the parameters are to be calculated. z=0.0 corresponds to the waist of the beam. 
        
    Returns: 
        w (double): The beam radius at the z position. 
        zR (double): The Rayleigh length of the beam (independent from z).
        R (double): The radius of curvature of the wavefront at the z position. At the waist (z=0.0) R=inf.
    """
    zR = np.pi * waist_radius**2 / wavelength
    w = waist_radius * np.sqrt(1 + (z / zR)**2)
    if (z != 0):
        R = z * (1 + (zR / z)**2)
    elif (z == 0):
        R = np.inf
    return w, zR, R

def laguerre_gauss(x, y, z, wavelength = 600e-9, waist_radius = 100e-6, radial_order = 0, azimuthal_order = 0):
    # create output
    lg_beam = np.zeros((np.shape(x)[0], np.shape(y)[0]), dtype = complex)
    
    # calculate wavenumber and beam parameters
    wavenumber = 2.0 * np.pi / wavelength
    w, zR, R = laguerre_gauss_paraxial_parameters(wavelength, waist_radius, radial_order, azimuthal_order, z)
    
    # generate correct laguerre polynomial
    laguerre_polynomial = genlaguerre(radial_order, azimuthal_order, monic = False)
    
    # generate field
    for ix in range (0, np.shape(x)[0]):
        for iy in range (0, np.shape(y)[0]):
            rho = np.sqrt(x[ix]**2 + y[iy]**2)
            phi = np.arctan2(y[iy], x[ix])
            term1 = 1 / w
            term2 = ((rho * np.sqrt(2)) / (w))**np.abs(azimuthal_order)
            term3 = np.exp(-(rho**2) / (w**2))
            term4 = laguerre_polynomial((2 * rho**2) / (w**2))
            term5 = np.exp(-1j * wavenumber * rho**2 / (2 * R))
            term6 = np.exp(-1j * azimuthal_order * phi)
            term7 = np.exp(1j * (np.abs(azimuthal_order) + 2 * radial_order + 1) * np.arctan(z/zR))
            lg_beam[ix, iy] = term1 * term2 * term3 * term4 * term5 * term6# * term7
    lg_beam /= np.max(np.abs(lg_beam))
    return lg_beam


    
    
