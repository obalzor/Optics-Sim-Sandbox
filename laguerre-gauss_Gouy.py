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
    radial_order = 2
    azimuthal_order = 2
    
    rayleigh = sources.laguerre_gauss_paraxial_parameters(wavelength, waist_radius, radial_order, azimuthal_order, 0.0)[1]
    
    print('Rayleigh length of beam:', rayleigh*1e3, 'mm')
    
    # sampling parameters
    frame = 2.0 * waist_radius
    n = 201
    
    window_x = 2.0 * (frame + waist_radius)
    x_min = -window_x / 2.0
    x_max = window_x / 2.0
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(x_min, x_max, n)
    
    # generate lg beam at waist
    lg_waist = sources.laguerre_gauss(x, y, 0.0, wavelength, waist_radius, radial_order, azimuthal_order)
    
    # display lg beam at waist
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (20, 20))
    cax_amplitude = ax1.imshow(np.abs(lg_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_amplitude)
    cbar1 = fig.colorbar(cax_amplitude)
    cbar1.set_label('Strength of electric field [a.u.]')
    ax1.set_xlabel('$x$ [m]')
    ax1.set_ylabel('$y$ [m]')
    cax_phase = ax2.imshow(np.angle(lg_waist), extent = (x_min, x_max, x_min, x_max), vmin = -np.pi, vmax= np.pi, cmap = cmap_phase)
    cbar2 = fig.colorbar(cax_phase)
    cbar2.set_label('Phase of electric field [rad]')
    ax2.set_xlabel('$x$ [m]')
    ax2.set_ylabel('$y$ [m]')
    cax_real = ax3.imshow(np.real(lg_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar3 = fig.colorbar(cax_real)
    ax3.set_xlabel('$x$ [m]')
    ax3.set_ylabel('$y$ [m]')
    cax_imaginary = ax4.imshow(np.imag(lg_waist), extent = (x_min, x_max, x_min, x_max), cmap = cmap_smooth)
    cbar4 = fig.colorbar(cax_imaginary)
    ax4.set_xlabel('$x$ [m]')
    ax4.set_ylabel('$y$ [m]')
    plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 0))
    plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 0))
    plt.tight_layout()
    plt.show()
    
    fig.savefig('results\\LG_Beam_' + str(radial_order) + '_' + str(azimuthal_order) + '_waist.png', dpi = 300, bbox_inches = 'tight')
    

    


