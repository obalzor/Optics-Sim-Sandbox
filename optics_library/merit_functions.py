# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:06:38 2024

@author: Myself
"""

import numpy as np

def centroid(field, dx, dy, x0 = 0.0, y0 = 0.0):
    nx = np.shape(field)[0]
    ny = np.shape(field)[1]
    windowX = (nx - 1) * dx
    windowY = (ny - 1) * dy
    xMin = x0 - windowX / 2
    xMax = x0 + windowX / 2
    yMin = y0 - windowY / 2
    yMax = y0 + windowY / 2
    
    x = np.linspace(xMin, xMax, nx)
    y = np.linspace(yMin, yMax, ny)
    x, y = np.meshgrid(x, y)
    
    sumSquaredAmplitude = np.sum((np.abs(field)) ** 2) * dx * dy
    sumWeightedSquaredAmplitudeX = np.sum(x * (np.abs(field) ** 2)) * dx * dy
    sumWeightedSquaredAmplitudeY = np.sum(y * (np.abs(field) ** 2)) * dx * dy
    
    centroidX = sumWeightedSquaredAmplitudeX / sumSquaredAmplitude
    centroidY = sumWeightedSquaredAmplitudeY / sumSquaredAmplitude
    
    return centroidX, centroidY

def integrateSquaredAmplitude(function, dx, dy):
    sum = 0.0
    sum += np.sum(np.abs(function) ** 2)
    sum *= (dx * dy)
    return sum

def beamSize(field, dx, dy, x0 = 0.0, y0 = 0.0):
    centroidX, centroidY = centroid(field, dx, dy, x0, y0)
    integralSquaredAmplitude = integrateSquaredAmplitude(field, dx, dy)
    nx = np.shape(field)[0]
    ny = np.shape(field)[1]
    windowX = (nx - 1) * dx
    windowY = (ny - 1) * dy
    xMin = x0 - windowX / 2
    xMax = x0 + windowX / 2
    yMin = y0 - windowY / 2
    yMax = y0 + windowY / 2
    x = np.linspace(xMin, xMax, nx)
    y = np.linspace(yMin, yMax, ny)
    numeratorX = np.sum((np.abs(field) ** 2) * ((x - centroidX) ** 2))
    numeratorX *= (dx * dy)
    numeratorX /= integralSquaredAmplitude
    numeratorX = 4.0 * np.sqrt(numeratorX)
    numeratorY = np.sum((np.abs(field) ** 2) * ((y - centroidY) ** 2))
    numeratorY *= (dx * dy)
    numeratorY /= integralSquaredAmplitude
    numeratorY = 4.0 * np.sqrt(numeratorY)
    return numeratorX, numeratorY


