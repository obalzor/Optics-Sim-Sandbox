# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:28:18 2024

@author: Myself
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from scipy.optimize import minimize
import optics_library.sources as sources
import optics_library.custom_fts as cust

if __name__ == '__main__':
    
    # basic parameters of the simulation
    debug = True
    
    cmap_amplitude = 'viridis'
    cmap_phase = 'Greys'
    cmap_smooth = 'jet'
    
    # parameters of source
    wavelength = 640e-9
    wavenumber = 2.0 * np.pi / wavelength
    
    # parameters for fundamental gaussian
    waist_radius_00 = 100e-6
    paraxial_parameters_00 = sources.laguerre_gauss_paraxial_parameters(wavelength, waist_radius_00, 0, 0, 0.0)
    zR_00 = paraxial_parameters_00[1]
    print('Rayleigh length of fundamental:', zR_00, 'm')
    
    # parameters of laguerre-gauss
    waist_radius_lg = waist_radius_00
    radial_order = 0
    azimuthal_order = 1
    paraxial_parameters_lg = sources.laguerre_gauss_paraxial_parameters(wavelength, waist_radius_lg, radial_order, azimuthal_order, 0.0)
    zR_lg = paraxial_parameters_lg[1]
    print('Rayleigh length of LG', radial_order, ',', azimuthal_order, ':', zR_lg, 'm')
    
    # sampling parameters for source
    window_x = 5e-3
    n = 201
    x_min = -window_x / 2.0
    x_max = window_x / 2.0
    dx = window_x / (n - 1)
    
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(x_min, x_max, n)
    
    gauss_waist = sources.laguerre_gauss(x, y, 0.0, wavelength, waist_radius_00, 0, 0)
    lg_waist = sources.laguerre_gauss(x, y, 0.0, wavelength, waist_radius_lg, radial_order, azimuthal_order)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))
    cax_gauss_amplitude = ax1.imshow(np.abs(gauss_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_amplitude)
    cbar1 = fig.colorbar(cax_gauss_amplitude)
    cbar1.set_label('Strength of eletric field [a.u.]')
    ax1.set_title('Amplitude of fundamental Gaussian at waist')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cax_gauss_phase = ax2.imshow(np.angle(gauss_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_phase)
    cbar2 = fig.colorbar(cax_gauss_phase)
    cbar2.set_label('Phase [rad]')
    ax2.set_title('Phase of fundamental Gaussian at waist')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    cax_lg_amplitude = ax3.imshow(np.abs(lg_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_amplitude)
    cbar3 = fig.colorbar(cax_lg_amplitude)
    cbar3.set_label('Strength of electric field [a.u.]')
    ax3.set_title('Amplitude of LG p='+str(radial_order)+',l='+str(azimuthal_order)+'at waist')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cax_lg_phase = ax4.imshow(np.angle(lg_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_phase)
    cbar4 = fig.colorbar(cax_lg_phase)
    cbar4.set_label('Phase [rad]')
    ax4.set_title('Phase of LG p='+str(radial_order)+',l='+str(azimuthal_order)+'at waist')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.ticklabel_format(axis = 'both', style = 'scientific', scilimits = (0,0))
    plt.tight_layout()
    plt.show()
    
    # construct shifted vortex
    vortex_waist = np.zeros((n, n), dtype = complex)
    for ix in range (0, n):
        for iy in range (0, n):
            vortex_waist[ix, iy] = gauss_waist[ix, iy] + lg_waist[ix, iy] 
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))
    cax_vortex_amplitude = ax1.imshow(np.abs(vortex_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_amplitude)
    cbar1 = fig.colorbar(cax_vortex_amplitude)
    cbar1.set_label('Strength of electric field [a.u.]')
    ax1.set_title('Amplitude of shifted vortex')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cax_vortex_phase = ax2.imshow(np.angle(vortex_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_phase)
    cbar2 = fig.colorbar(cax_vortex_phase)
    cbar2.set_label('Phase [rad]')
    ax2.set_title('Phase of shifted vortex')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    cax_vortex_real = ax3.imshow(np.real(vortex_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar3 = fig.colorbar(cax_vortex_real)
    cbar3.set_label('Strength of electric field [a.u.]')
    ax3.set_title('Real part of shifted vortex')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cax_vortex_imaginary = ax4.imshow(np.imag(vortex_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar4 = fig.colorbar(cax_vortex_imaginary)
    cbar4.set_label('Strength of electric field [a.u.]')
    ax4.set_title('Imaginary part of shifted vortex')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.ticklabel_format(axis = 'both', style = 'scientific', scilimits = (0,0))
    plt.tight_layout()
    plt.show()
    
    # propagate shifted vortex 
    nRayleigh = 10
    z_min = -nRayleigh * zR_00
    z_max = nRayleigh * zR_00
    nz = 51
    dz = (z_max - z_min) / (nz - 1)
    z = np.linspace(z_min, z_max, nz)
    
    # generate documents for animation
    propagated_shifted_vortex = []
    propagated_shifted_vortex_k = []
    gouy_gauss = np.zeros(n)
    gouy_lg = np.zeros(n)
    
    for iz in range (0, nz):
        propagated_shifted_vortex.append(sources.laguerre_gauss(x, y, z[iz], wavelength, waist_radius_00, 0, 0) + sources.laguerre_gauss(x, y, z[iz], wavelength, waist_radius_lg, radial_order, azimuthal_order))
        propagated_shifted_vortex_k.append(cust.fft(propagated_shifted_vortex[iz], dx, dx))
        gouy_gauss[iz] = np.arctan(z/zR_00)
        gouy_lg[iz] = (1 + np.abs(azimuthal_order) + 2 * radial_order) * np.arctan(z/zR_lg)
        
        fig, (ax1, ax2
    
        
    
        
    
    
    
    
    
    
    
    



