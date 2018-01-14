# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:26:34 2017

@author: cedric
"""
from __future__ import print_function
from PIL import Image 


Im = Image.open("cat.jpg")
print (Im.mode,Im.size,Im.format)

Im.show()

#用image模块实现图片的参数显示


newIm = Image.new ("RGBA", (640, 480), (255, 0, 0))