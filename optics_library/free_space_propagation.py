# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:12:45 2024

@author: Myself
"""

import numpy as np
import custom_fts as cust

def spw_propagation(field, dx, dy, wavelength, propagation_distance):
    # calculate sampling parameters
    nx = np.shape(field)[0]
    ny = np.shape(field)[1]
    windowX = (nx - 1) * dx
    windowY = (ny - 1) * dy
    
    windowKx = 2 * np.pi / dx
    windowKy = 2 * np.pi / dy
    dkx = 2 * np.pi / windowX
    dky = 2 * np.pi / windowY
    kxMin = -windowKx / 2
    kyMin = -windowKy / 2
    
    wavenumber = 2 * np.pi / wavelength
    
    # perform ft
    spectrum = cust.fft(field, dx, dy)
    
    # propagate spectrum
    for ix in range (0, nx):
        kx = kxMin + ix * dkx
        for iy in range (0, ny):
            ky = kyMin + iy * dky
            kz = np.sqrt(wavenumber ** 2 - (kx ** 2 + ky ** 2))
            spectrum[ix, iy] *= np.exp(1j * kz * propagation_distance)
    
    # perform inverse ft
    propagated_field = cust.ifft(spectrum, dkx, dky)
    return propagated_field

def rayleigh_sommerfeld(field, dx, dy, propagation_distance, wavelength, x0_prop, y0_prop, windowX_prop, windowY_prop, nx_prop, ny_prop):
    nx = np.shape(field)[0]
    ny = np.shape(field)[1]
    windowX = (nx - 1) * dx
    windowY = (ny - 1) * dy
    xMin = -windowX / 2
    yMin = -windowY / 2
    
    xMin_prop = x0_prop - windowX_prop / 2
    yMin_prop = y0_prop - windowY_prop / 2
    dx_prop = windowX_prop / (nx_prop - 1)
    dy_prop = windowY_prop / (ny_prop - 1)
    
    propagated_field = np.zeros((nx_prop, ny_prop), dtype = complex)
    wavenumber = 2 * np.pi / wavelength
    
    for ix_prop in range (0, nx_prop):
        x_prop = xMin_prop + ix_prop * dx_prop
        for iy_prop in range (0, ny_prop):
            y_prop = yMin_prop + iy_prop * dy_prop
            for ix in range (0, nx):
                x = xMin + ix * dx
                for iy in range (0, ny):
                    y = yMin + iy * dy
                    r = np.sqrt((x - x_prop) ** 2 + (y - y_prop) ** 2 + (propagation_distance) ** 2)
                    propagated_field[ix_prop, iy_prop] += field[ix, iy] * ((1/ wavenumber * r) - 1j) * (propagation_distance / r) * (np.exp(1j * wavenumber * r) / r)
            propagated_field[ix_prop, iy_prop] *= (1 / wavelength) * dx * dy
    return propagated_field



