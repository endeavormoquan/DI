# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:53:09 2017

@author: Cedric
"""

from PIL import Image 
import glob, os 

size = 128, 128 

for infile in glob.glob("*.jpg"): 
    file, ext = os.path.splitext(infile) 
    im = Image.open(infile) 
    im.thumbnail(size, Image.ANTIALIAS) 
    im.save(file + ".thumbnail", "JPEG") 