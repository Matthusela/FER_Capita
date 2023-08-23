# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 22:59:44 2023

@author: matt8
"""


# This file is set up to be run from a python enabled command line, this file simply activates the Model_Files() function and begins the process of
# fitting models with functionality to configure one-hot and GreyScale/RGB settings. To run in spyder just remove all of the arg parser lines and specify
# oneHot and RGB_or_GS in spyder as Booleans

import argparse
from RandomModels import Model_Files

# Create an argument parser
parser = argparse.ArgumentParser()

# Add boolean arguments
parser.add_argument("--OneHot", action="store_true",default=False, help="Enable OneHot")
parser.add_argument("--RGB", action="store_true",default=False, help="Enable RGB")

# Parse the command-line arguments
args = parser.parse_args()

# Access the boolean values
oneHot = args.OneHot
RGB_or_GS = args.RGB

Model_Files(oneHot, RGB_or_GS)
