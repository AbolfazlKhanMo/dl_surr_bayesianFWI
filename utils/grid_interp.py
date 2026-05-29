#!/usr/bin/env python
# coding: utf-8

# import libraries
import numpy as np
from scipy.interpolate import griddata

from config import resX, resY


# Function for interpolation from unstructured to structured
def Grid(x, y, z, resX=resX, resY=resY):
    """
    Converts 3 column data to matplotlib grid
    """

    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)

    # scipy version
    Z = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear', fill_value=0.0)

    return Z