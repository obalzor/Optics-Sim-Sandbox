# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:09:25 2024

@author: Myself
"""

import numpy as np

def deviation(reference, data, dx, dy, allow_scaling):
    if (np.shape(data) != np.shape(reference)):
        raise ('Both the data set for analysis and the reference need to have the same number of pixels. Sampling assumed equidistant.')
        return
    
    if (allow_scaling):
        dataConj= np.conjugate(data)
        
        sumSquaredData = np.sum(np.abs(data) ** 2) * dx * dy
        sumProjection = np.sum(dataConj * reference) * dx * dy
        sumSquaredReference = np.sum(np.abs(reference) ** 2) * dx * dy
        
        scalingFactor = sumProjection / sumSquaredData
        
        sumSquaredDifference = np.sum(np.abs((scalingFactor * data) - reference) ** 2) * dx * dy
        
        absoluteDeviation = sumSquaredDifference 
        relativeDeviation = sumSquaredDifference / sumSquaredReference
        
        
    elif (allow_scaling == False):
        sumSquaredReference = np.sum(np.abs(reference) ** 2) * dx * dy
        sumSquaredDifference = np.sum(np.abs(data - reference) ** 2) * dx * dy
        
        absoluteDeviation = sumSquaredDifference
        relativeDeviation = sumSquaredDifference / sumSquaredReference
        scalingFactor = 1.0 * 0.0j
        
        
    return absoluteDeviation, relativeDeviation, scalingFactor