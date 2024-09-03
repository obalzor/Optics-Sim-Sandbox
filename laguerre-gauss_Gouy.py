# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:38:12 2024

@author: Myself
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import genlaguerre
from scipy.optimize import minimize
import optics_library.sources as sources
import optics_library.custom_fts as cust

if __name__ == "__main__": 
    # basic parameters of simulation
    debug = True
    
    cmap_amplitude = 'viridis'
    cmap_phase = 'Greys'
    cmap_smooth = 'jet'
    
    # parameters of source
    wavelength = 640e-9
    wavenumber = 2.0 * np.pi / wavelength
    waist_radius = 100e-6
    radial_order = 0
    azimuthal_order = 1
    
    rayleigh = sources.laguerre_gauss_paraxial_parameters(wavelength, waist_radius, radial_order, azimuthal_order, 0.0)[1]
    
    print('Rayleigh length of beam:', rayleigh*1e3, 'mm')
    
    # sampling parameters
    frame = 5.0 * waist_radius
    n = 201
    
    window_x = 2.0 * (frame + waist_radius)
    x_min = -window_x / 2.0
    x_max = window_x / 2.
    dx = window_x / (n - 1)
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(x_min, x_max, n)
    
    # generate lg beam at waist
    lg_waist = sources.laguerre_gauss(x, y, 0.0, wavelength, waist_radius, radial_order, azimuthal_order)
    gauss_waist = sources.laguerre_gauss(x, y, 0.0, wavelength, waist_radius, 0, 0)
    shifted_vortex_waist = lg_waist + gauss_waist

    # display lg beam at waist
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))
    cax_amplitude = ax1.imshow(np.abs(lg_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_amplitude)
    cbar1 = fig.colorbar(cax_amplitude)
    cbar1.set_label('Strength of electric field [a.u.]')
    ax1.set_title('Amplitude of LG beam ' + str(radial_order) + ', ' + str(azimuthal_order) + ' at waist')
    ax1.set_xlabel('$x$ [m]')
    ax1.set_ylabel('$y$ [m]')
    cax_phase = ax2.imshow(np.angle(lg_waist), extent = (x_min, x_max, x_min, x_max), vmin = -np.pi, vmax= np.pi, cmap = cmap_phase)
    cbar2 = fig.colorbar(cax_phase)
    cbar2.set_label('Phase of electric field [rad]')
    ax2.set_title('Phase of LG beam ' + str(radial_order) + ', ' + str(azimuthal_order) + ' at waist')
    ax2.set_xlabel('$x$ [m]')
    ax2.set_ylabel('$y$ [m]')
    cax_real = ax3.imshow(np.real(lg_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar3 = fig.colorbar(cax_real)
    ax3.set_title('Real part of LG beam ' + str(radial_order) + ', ' + str(azimuthal_order) + ' at waist')
    ax3.set_xlabel('$x$ [m]')
    ax3.set_ylabel('$y$ [m]')
    cax_imaginary = ax4.imshow(np.imag(lg_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar4 = fig.colorbar(cax_imaginary)
    ax4.set_title('Imaginary part of LG beam ' + str(radial_order) + ', ' + str(azimuthal_order) + ' at waist')
    ax4.set_xlabel('$x$ [m]')
    ax4.set_ylabel('$y$ [m]')
    plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 0))
    plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 0))
    plt.tight_layout()
    plt.show()
    
    #fig.savefig('results\\LG_Beam_' + str(radial_order) + '_' + str(azimuthal_order) + '_waist.png', dpi = 300, bbox_inches = 'tight')
    
    # display coherent sum of modes (shifted vortex)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))
    cax_amplitude = ax1.imshow(np.abs(shifted_vortex_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_amplitude)
    cbar1 = fig.colorbar(cax_amplitude)
    cbar1.set_label('Strength of electric field [a.u.]')
    ax1.set_title('Amplitude of shifted vortex at waist')
    ax1.set_xlabel('$x$ [m]')
    ax1.set_ylabel('$y$ [m]')
    cax_phase = ax2.imshow(np.angle(shifted_vortex_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_phase)
    cbar2 = fig.colorbar(cax_phase)
    cbar2.set_label('Phase of electric field [rad]')
    ax2.set_title('Phase of shifted vortex at waist')
    ax2.set_xlabel('$x$ [m]')
    ax2.set_ylabel('$y$ [m]')
    cax_real = ax3.imshow(np.real(shifted_vortex_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar3 = fig.colorbar(cax_real)
    cbar3.set_label('Real part of electric field [a.u.]')
    ax3.set_title('Real part of shifted vortex at waist')
    ax3.set_xlabel('$x$ [m]')
    ax3.set_ylabel('$y$ [m]')
    cax_imaginary = ax4.imshow(np.imag(shifted_vortex_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar4 = fig.colorbar(cax_imaginary)
    cbar4.set_label('Imaginary part of electric field [a.u.]')
    ax4.set_title('Imaginary part of shifted vortex at waist')
    ax4.set_xlabel('$x$ [m]')
    ax4.set_ylabel('$y$ [m]')
    plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 0))
    plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 0))
    plt.tight_layout()
    plt.show()
    
    #fig.savefig('results\\shifted_vortex_waist.png', dpi = 300, bbox_inches = 'tight')
    
    # calculate spectrum of shifted vortex
    shifted_vortex_waist_spectrum = cust.fft(shifted_vortex_waist, dx, dx)
    window_k = 2.0 * np.pi / dx
    k_min = -window_k / 2.0
    k_max = window_k / 2.0
    dk = 2.0 * np.pi / window_x
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))
    cax_amplitude = ax1.imshow(np.abs(shifted_vortex_waist_spectrum), extent = (k_min, k_max, k_min, k_max), cmap = cmap_amplitude)
    cbar1 = fig.colorbar(cax_amplitude)
    cbar1.set_label('Amplitude of $k$ spectrum of electric field [a.u.]')
    ax1.set_title('Amplitude of spectrum of shifted vortex at waist')
    ax1.set_xlabel('$k_x$ [1/m]')
    ax1.set_ylabel('$k_y$ [1/m]')
    cax_phase = ax2.imshow(np.angle(shifted_vortex_waist_spectrum), extent = (k_min, k_max, k_min, k_max), cmap = cmap_phase)
    cbar2 = fig.colorbar(cax_phase)
    ax2.set_title('Phase of spectrum of shifted vortex at waist')
    ax2.set_xlabel('$k_x$ [1/m]')
    ax2.set_ylabel('$k_y$ [1/m]')
    cax_real = ax3.imshow(np.real(shifted_vortex_waist_spectrum), extent = (k_min, k_max, k_min, k_max), cmap = cmap_smooth)
    cbar3 = fig.colorbar(cax_real)
    cbar3.set_label('Real part of electric field [a.u.]')
    ax3.set_title('Real part of spectrum of shifted vortex at waist')
    ax3.set_xlabel('$k_x$ [1/m]')
    ax3.set_ylabel('$k_y$ [1/m]')
    cax_imaginary = ax4.imshow(np.imag(shifted_vortex_waist_spectrum), extent = (k_min, k_max, k_min, k_max), cmap = cmap_smooth)
    cbar4 = fig.colorbar(cax_imaginary)
    cbar4.set_label('Imaginary part of electric field [a.u.]')
    ax4.set_title('Imaginary part of spectrum of shifted vortex at waist')
    ax4.set_xlabel('$k_x$ [1/m]')
    ax4.set_ylabel('$k_y$ [1/m]')
    plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 0))
    plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 0))
    plt.tight_layout()
    plt.show()
    
    # define propagation parameters
    rayleigh_factor = 5.0
    z_min = -rayleigh_factor * rayleigh
    z_max = rayleigh_factor * rayleigh
    nz = 51
    dz = (z_max - z_min) / (nz - 1)
    z = np.linspace(z_min, z_max, nz)
    
    # generate propagation data
    shifted_vortex = []
    
    for iz in range (0, nz): 
        shifted_vortex.append(sources.laguerre_gauss(x, y, z[iz], wavelength, waist_radius, radial_order, azimuthal_order) + sources.laguerre_gauss(x, y, z[iz], wavelength, waist_radius, 0, 0))
        
    # set up figure for animation
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))
    cax_amplitude = ax1.imshow(np.abs(shifted_vortex[0]), extent = (x_min, x_max, x_min, x_max), cmap = cmap_amplitude)
    cbar1 = fig.colorbar(cax_amplitude)
    cbar1.set_label('Strength of electric field [a.u.]')
    ax1.set_title('Amplitude of shifted vortex')
    ax1.set_xlabel('$x$ [m]')
    ax1.set_ylabel('$y$ [m]')
    cax_phase = ax2.imshow(np.angle(shifted_vortex[0]), extent = (x_min, x_max, x_min, x_max), cmap = cmap_phase)
    cbar2 = fig.colorbar(cax_phase)
    cbar2.set_label('Phase [rad]')
    ax2.set_title('Phase of shifted vortex')
    ax2.set_xlabel('$x$ [m]')
    ax2.set_ylabel('$y$ [m]')
    cax_real = ax3.imshow(np.real(shifted_vortex[0]), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar3 = fig.colorbar(cax_real)
    cbar3.set_label('Strength of electric field [a.u.]')
    ax3.set_title('Real part of shifted vortex')
    ax3.set_xlabel('$x$ [m]')
    ax3.set_ylabel('$y$ [m]')
    cax_imaginary = ax4.imshow(np.imag(shifted_vortex[0]), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar4 = fig.colorbar(cax_imaginary)
    cbar4.set_label('Strength of electric field [a.u]')
    ax4.set_title('Imaginary part of shifted vortex')
    ax4.set_xlabel('$x$ [m]')
    ax4.set_ylabel('$y$ [m]')
    plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 0))
    plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 0))
    plt.tight_layout()
    plt.show()
    
    # create animation
    def update_frame(i):
        cax_amplitude.set_array(np.abs(shifted_vortex[i]))
        cax_phase.set_array(np.angle(shifted_vortex[i]))
        cax_real.set_array(np.real(shifted_vortex[i]))
        cax_imaginary.set_array(np.imag(shifted_vortex[i]))
        return cax_amplitude, cax_phase, cax_real, cax_imaginary
    
    animation_shifted_vortex = animation.FuncAnimation(fig, update_frame, frames = nz, interval = 50, blit = True)
    animation_shifted_vortex.save('results\\animation_shifted_vortex_withoutGouy.gif', writer = 'imagemagick')
    
    
    # plot Gouy phase according to paraxial formula
    
    
    

    


