# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:39:55 2024
Generate an animation of a Laguerre-Gauss beam as it propagates through the waist. 
Use the formulas for paraxial propagation. 
@author: Myself
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import genlaguerre
from scipy.optimize import minimize
import optics_library.custom_fts as cust
import optics_library.sources as sources

if __name__ == "__main__":
    # basic parameters of the simulation
    debug = True

    cmap_amplitude = 'viridis'
    cmap_phase = 'Greys'
    cmap_smooth = 'jet'

    # parameters of source
    wavelength = 600e-3
    wavenumber = 2.0 * np.pi / wavelength
    waist_radius = 100e-6
    radial_order = 0
    azimuthal_order = 1

    # sampling parameters
    n = 201
    frame = 20 * waist_radius

    window_x = 2.0 * (frame + waist_radius)
    x_min = -window_x / 2.0
    x_max = window_x / 2.0

    x = np.linspace(x_min, x_max, n)
    y = np.linspace(x_min, x_max, n) # equal sampling in x and y axes.
    #x,y = np.meshgrid(x, y)

    # beam at waist
    lg_waist = sources.laguerre_gauss(x, y, 0.0, wavelength, waist_radius, radial_order, azimuthal_order)
    w0, rayleigh, R0 = sources.laguerre_gauss_paraxial_parameters(wavelength, waist_radius, radial_order, azimuthal_order, 0.0)

    print('Waist Radius:', w0, 'm')
    print('Radius of Curvature of Wavefront:', R0, 'm')
    print('Rayleigh length:', rayleigh, 'm')

    fig, ax = plt.subplots(2, 1, sharex = True, sharey = True, figsize = (20, 10))
    cax_amplitude = ax[0].imshow(np.abs(lg_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_amplitude)
    cbar_amplitude = fig.colorbar(cax_amplitude)
    cbar_amplitude.set_label('Amplitude of Electric Field [a.u.]')
    ax[0].set_title('Amplitude of Laguerre-Gauss ' + str(radial_order) + ', ' + str(azimuthal_order) + ' at waist')
    ax[0].set_xlabel('$x$ [m]')
    ax[0].set_ylabel('$y$ [m]')
    cax_phase = ax[1].imshow(np.angle(lg_waist), extent = (x_min, x_max, x_min, x_max), vmin = -np.pi, vmax = np.pi, cmap = cmap_phase)
    cbar_phase = fig.colorbar(cax_phase)
    cbar_phase.set_label('Phase [rad]')
    ax[1].set_title('Phase of Laguerre-Gauss ' + str(radial_order) + ', ' + str(azimuthal_order) + ' at waist')
    ax[1].set_xlabel('$x$ [m]')
    ax[1].set_ylabel('$y$ [m]')
    plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0,0))
    plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0))
    plt.tight_layout()
    plt.show()

    fig.savefig('results\\LG beam ' + str(radial_order) + ' ' + str(azimuthal_order) + ' waist.png', dpi= 300, bbox_inches = 'tight')
    
    # propagation
    zR_factor = 10.0
    z_min = -zR_factor * rayleigh
    z_max = zR_factor * rayleigh
    nz = 101
    z = np.linspace(z_min, z_max, nz)
    
    prop_amplitude = []
    prop_phase = []
    prop_real = []
    
    for iz in range (0, nz):
        field = sources.laguerre_gauss(x, y, z[iz], wavelength, waist_radius, radial_order, azimuthal_order)
        prop_amplitude.append(np.abs(field))
        prop_phase.append(np.angle(field))
        prop_real.append(np.real(field))
        
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, sharey = True, figsize = (10, 20))
    im1 = ax1.imshow(prop_amplitude[0], extent = (x_min, x_max, x_min, x_max), cmap = cmap_amplitude, animated = True)
    im2 = ax2.imshow(prop_phase[0], extent = (x_min, x_max, x_min, x_max), vmin = -np.pi, vmax= np.pi, cmap = cmap_phase, animated = True)
    
    cbar_amplitude = fig.colorbar(im1, ax = ax1)
    cbar_amplitude.set_label('Amplitude of Electric Field [a.u.]')
    cbar_phase = fig.colorbar(im2, ax = ax2)
    cbar_phase.set_label('Phase [rad]')
    
    ax1.set_title('Real Part of Laguerre-Gauss ' + str(radial_order) + ', ' + str(azimuthal_order))
    ax2.set_title('Phase of Laguerre-Gauss ' + str(radial_order) + ', ' + str(azimuthal_order))
    ax1.set_xlabel('$x$ [m]')
    ax2.set_xlabel('$x$ [m]')
    ax1.set_ylabel('$y$ [m]')
    ax2.set_ylabel('$y$ [m]')
    plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0,0))
    plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0))
    plt.tight_layout()
    plt.show()
    
    
    
    def update_frame(i):
        im1.set_array(prop_real[i])
        im2.set_array(prop_phase[i])
        return im1, im2
    
    animation_amplitude_phase = animation.FuncAnimation(fig, update_frame, frames = nz, interval = 50, blit = True)
    animation_amplitude_phase.save('results\\Animation RePa ' + str(radial_order) + ', ' + str(azimuthal_order) + '.gif', writer = 'imagemagick')
    
    

    



