# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:28:26 2024

@author: Myself
"""

import numpy as np

def fft(field, dx, dy):
    nx = np.shape(field)[0]
    ny = np.shape(field)[1]
    windowX = (nx - 1) * dx
    windowY = (ny - 1) * dy
    
    dkx = 2.0 * np.pi / windowX
    dky = 2.0 * np.pi / windowY
    windowKx = 2.0 * np.pi / dx
    windowKy = 2.0 * np.pi / dy
    kxMin = -windowKx / 2.0
    kyMin = -windowKy / 2.0
    
    spectrum = np.fft.fft2(field)
    spectrum = np.fft.fftshift(spectrum)
    spectrum *= (dx * dy / (2.0 * np.pi))
    
    for ix in range (0, nx):
        coordKx = kxMin + ix * dkx
        coordKx /= windowKx
        for iy in range (0, ny):
            coordKy = kyMin + iy * dky
            coordKy /= windowKy
            spectrum[ix, iy] *= np.exp(1j * np.pi * (ix + iy - coordKx - coordKy))
            
    return spectrum

def ifft(spectrum, dkx, dky):
    nx = np.shape(spectrum)[0]
    ny = np.shape(spectrum)[1]
    windowKx = (nx - 1) * dkx
    windowKy = (ny - 1) * dky
    
    dx = 2.0 * np.pi / windowKx
    dy = 2.0 * np.pi / windowKy
    windowX = 2.0 * np.pi / dkx
    windowY = 2.0 * np.pi / dky
    xMin = -windowX / 2.0
    yMin = -windowY / 2.0
    
    field = np.fft.ifft2(spectrum)
    field = np.fft.fftshift(field)
    field *= (dkx * dky * nx * ny / (2.0 * np.pi))
    
    for ix in range (0, nx):
        coordX = xMin + ix * dx
        coordX /= windowX
        for iy in range (0, ny):
            coordY = yMin + iy * dy
            coordY /= windowY
            field[ix, iy] *= np.exp(-1j * np.pi * (ix + iy - coordX - coordY))
            
    return field


