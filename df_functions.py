# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:09:18 2023

@author: matt8
"""
import numpy as np
import pandas as pd
import math 
# Quickly defining two functions to help ensure images have been pixelated properly
def image_dimension(data_frame):
    pxls = data_frame.pixels_list.apply(len)
    if pxls.isin(np.array(pxls)).all() != True:
        return "Image pixelation is not consistent across images"
    else:
        length = np.array(pxls,dtype =np.uint16)[0]
        dim = math.sqrt(length)
        if dim%1 == 0:
            return int(dim)
        else:
            return "Image pixelation has not produced a square image"
        
def number_of_images(data_frame):
    return len(data_frame.index)