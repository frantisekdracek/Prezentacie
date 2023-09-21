#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def RBG_to_HSV(r,g,b):
    R = r/255
    G = g/255
    B = b/255
    
    C_max = max(R, G, B)
    C_min = min(R, G, B)
    delta = C_max - C_min
    
    V = C_max
    
    if C_max == 0:
        S = 0
    else:
        S = delta/C_max
        
    if delta == 0:
        H = 0
    elif C_max == R:
        H = 60* (0 + (G-B)/delta)
    elif C_max == G:
        H = 60* (2 + (B-R)/delta)
    else:
        H = 60* (4 + (R-G)/delta)
    
    if H<0:
        H=H+360
        
    return H, S, V


print(RBG_to_HSV(100, 100, 100))



