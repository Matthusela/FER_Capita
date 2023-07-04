# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 22:59:44 2023

@author: matt8
"""



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
