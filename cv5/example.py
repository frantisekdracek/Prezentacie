#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:09:22 2023

@author: frantisekdracek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:12:27 2023

@author: frantisekdracek
"""

import pygame
import numpy as np

#resolution
WIDTH = 600
HEIGHT = 600

#colors
WHITE = (255, 255, 255)
BLUE = (20, 120, 240)
RED = (240, 120, 20)

def R_x(theta, homogenous = True):
    if homogenous:
        Rx = np.array([[1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
            ])        
    else:    
        Rx = np.array([[1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]])
    return Rx

def R_y(theta, homogenous = True):
    if homogenous:
        Ry = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                  [0, 1, 0, 0],
                  [-np.sin(theta), 0, np.cos(theta), 0],
                  [0, 0, 0, 1]
                  ])
    else:    
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    return Ry


def R_z(theta, homogenous = True):
    if homogenous:
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                  [np.sin(theta), np.cos(theta), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]
                  ])
    else:
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return Rz

def projection_matrix_simple_perspective_with_culling(n, f):
    P = np.array([
        [n, 0, 0, 0],
        [0, n, 0, 0],
        [0, 0, (f + n)/(f-n),  -2 * f * n/(f-n)],
        [0, 0, 1, 0]
        ])
    return P

def translation(t):
    T = np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, 1]
        ])
    return T

def rotation(theta):
    R = R_x(theta[0]) @ R_y(theta[1]) @ R_z(theta[2])
    return R

def rotation_from_vectors(direction, up):
    z_axis =  direction/np.linalg.norm(direction)
    up =  up/np.linalg.norm(up)
    x_axis = np.cross(z_axis, up)
    x_axis =  x_axis/np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, z_axis)
    y_axis =  y_axis/np.linalg.norm(y_axis)
    
    W = np.array([
        [x_axis[0], x_axis[1], x_axis[2], 0],
        [y_axis[0], y_axis[1], y_axis[2], 0],
        [z_axis[0], z_axis[1], z_axis[2], 0],
        [0, 0, 0, 1]
        ])
    return W

    
def scaling(s_x, s_y, t_x, t_y):
    S = np.array([
        [s_x, 0, t_x, 0],
        [0, s_y, t_y, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])
    return S


def perspective_projection(
        a, #point
        c, #camera position
        theta, #camera orientation
        e # screen coordiante frame origin
        ):
    
    a = np.concatenate((a, np.array([1])))
    R = rotation(theta)
    
    T = translation(-c)
    
    P = projection_matrix_simple_perspective_with_culling(0.3, 1000)
    
    f = P@R@T@a
    
    f = f/f[3]
    
    return f



class Cube:
    def __init__(self):
        self.vertices=np.array([
                 [-1,  1,  1],
                 [ 1,  1,  1],
                 [ 1, -1,  1],
                 [-1, -1,  1],
                 [-1,  1, -1],
                 [ 1,  1, -1],
                 [ 1, -1, -1],
                 [-1, -1, -1]           
                ])
        self.colors = np.array([
            [250, 0, 0],
            [240, 120, 20],
            [0, 250, 0],
            [0, 0, 250],
            [250, 250, 0],
            [250, 0, 250],
            [0, 250, 250],
            [20, 120, 240]
            ])
        self.primitives=np.array([
            [0, 1, 2], [2, 3, 1], 
            [4, 5, 6], [6, 7, 4],
            [0, 1, 5], [5, 4, 0],
            [3, 2, 6], [6, 7, 3],
            [0, 4, 7], [7, 3, 0],
            [1, 5, 6], [6, 2, 1]
        ])
        
    
    


cube = Cube()

c = np.array([0,0,-20]).T

theta = np.array([0,0,0]).T

e = np.array([WIDTH/2, HEIGHT/2, 20]).T


# configuration of pygame window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Display")
clock = pygame.time.Clock()
fps = 10


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
            
    # Clear the screen
    screen.fill(WHITE)
    
    #c = R_y(np.pi/100, homogenous=False)@c
    #direction = -c
    #cube.transform(R_y(np.pi/180, homogenous = False))
    #cube.transform(R_x(np.pi/180, homogenous = False))

    #R = rotation_from_vectors(direction, np.array([0, -1, 0]))
    #p = [perspective_projection_2(point, c, R, e) for point in cube.vertices]
    p = [perspective_projection(point, c, theta, e) for point in cube.vertices]
    
    p = [scaling(1000,1000, WIDTH/2, HEIGHT/2)@point for point in p]
    
    for point in p:
        pygame.draw.circle(screen, BLUE, (point[0], point[1]), 3)
        
    for i in range(cube.primitives.shape[0]):
        primitive = cube.primitives[i]
        p0 = cube.vertices[primitive[0]]
        p1 = cube.vertices[primitive[1]]
        p2 = cube.vertices[primitive[2]]
        
        print(p0)
        
        
        
        color = ((cube.colors[primitive[0]] + cube.colors[primitive[1]] + cube.colors[primitive[2]])/3).astype(int)
        pygame.draw.polygon(screen, color, [[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]]] )







    # Update the display
    pygame.display.update()
    clock.tick(fps)

# Quit Pygame
pygame.quit()