# Authors: Stefanos Papanikolaou <stephanos.papanikolaou@gmail.com>
# BSD 2-Clause License
# Copyright (c) 2021, PapStatMechMat
# All rights reserved.
# How to cite SeaPy:
# S. Papanikolaou, Data-Rich, Equation-Free Predictions of Plasticity and Damage in Solids, (under review in Phys. Rev. Materials) arXiv:1905.11289 (2019)

import cv2
import sys
#test
import pylab as plt
import importlib as imp
from numpy import asarray
import numpy as np
from PIL import Image
import webcolors

def ScalarImgCorrelations(img_arr,bins=2):
    img_max=img_arr.flatten().max()
    img_min=img_arr.flatten().min()
    img_bins=np.linspace(img_min,img_max,bins)
    print(img_bins)
    return None
    

def ImgToScalarArray(file):
    color_img=asarray(Image.open(file))/255.
    img_arr=np.mean(color_img,axis=2)
    return img_arr

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def GetMainColors(file):
    import extcolors
    img = Image.open(file)
    colors,pix_cnt = extcolors.extract_from_image(img) #put a higher value if there are m
    closest_colors=[]
    for c,d in colors:
        closest_colors.append(get_colour_name(c)[1])
    if len(colors)>5:
        return colors[:5],closest_colors[:5]
    else:
        return colors,closest_colors

def ReadImage(file):
    return Image.open(file)

def ReadImageToArray(file):
    img=cv2.imread(file)
    return img

def CropImage(img, percent_x,percent_y):
    if percent_x<0 or percent_x>100:
        sys.exit("Cropped Image percentages need to be between 0 and 100")
    if percent_y<0 or percent_y>100:
        sys.exit("Cropped Image percentages need to be between 0 and 100")    
    height,width=img.shape[0:2]
    startRow = int(height*percent_y/2./100.)
    startCol = int(width*percent_x/2./100.)
    endRow = int(height*(100.-percent_y/2.)/100.)
    endCol = int(width*(100.-percent_x/2.)/100.)
    return img[startRow:endRow, startCol:endCol]

def ImgToArray(img):
    return asarray(img)

def ImgToScalArr(img):
    return np.array(img.convert('L'))
    
    


