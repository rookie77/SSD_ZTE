#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:35:59 2017

@author: ra
"""

import math

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def hsv2colour( h, s, v):
    h=h/2;s=s*255; v=v*255
    colour='青色'
    if(h>=0 and h<=180 and s>=0 and s<=255 and v>=0 and v<=46):
        colour='黑色'
    elif(h>=0 and h<=180 and s>=0 and s<=43 and v>=46 and v<=220):
        colour='灰色'
    elif(h>=0 and h<=180 and s>=0 and s<=30 and v>=221 and v<=255):
        colour='白色'
    elif(h>=0 and h<=10 and s>=43 and s<=255 and v>=46 and v<=255):
        colour='红色'
    elif(h>=156 and h<=180 and s>=43 and s<=255 and v>=46 and v<=255):
        colour='红色'
    elif(h>=11 and h<=25 and s>=43 and s<=255 and v>=46 and v<=255):
        colour='橙色'
    elif(h>=26 and h<=34 and s>=43 and s<=255 and v>=46 and v<=255):
        colour='黄色'
    elif(h>=35 and h<=77 and s>=43 and s<=255 and v>=46 and v<=255):
        colour='绿色'  
    elif(h>=78 and h<=99 and s>=43 and s<=255 and v>=46 and v<=255):
        colour='青色'
    elif(h>=100 and h<=124 and s>=43 and s<=255 and v>=46 and v<=255):
        colour='蓝色'        
    elif(h>=125 and h<=155 and s>=43 and s<=255 and v>=46 and v<=255):
        colour='紫色'    
    return colour
