#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:12:27 2023

@author: frantisekdracek
"""

import pygame
import numpy as np

#resolution
WIDTH = 800
HEIGHT = 800

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


def projection_matrix_simple_perspective(d):
    #todo 
    #return P
    pass


def translation(t):
    #todo
    T = None
    return T

    
def scaling(s_x, s_y, t_x, t_y):
    #todo
    #return S
    pass

def perspective_projection(
        a, #point
        c, #camera position
        theta, #camera orientation
        e # screen coordiante frame origin
        ):
    
    #todo
    #return f
    pass

    
 
    


class Cube:
    def __init__(self):
        self.vertices = np.array([
                 [ 1,  1,  1],
                 [ 1,  1, -1],
                 [ 1, -1,  1],
                 [ 1, -1, -1],
                 [-1, -1, -1],
                 [-1,  1, -1],
                 [-1,  1,  1],
                 [-1, -1,  1]
                ])
        self.edges = np.array([[0, 1], [0, 2], [0, 6],
                     [1, 3], [1, 5], [2, 7],
                     [2, 3], [4, 7], [4, 3],
                     [4, 5], [5, 6], [6, 7]
                     ])
    def transform(self, A):
        self.vertices = self.vertices@A.T
    


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
    

    #p = [perspective_projection(point, c, R, e) for point in cube.vertices]
    #p = [scaling(100,100, WIDTH/2, HEIGHT/2)@point for point in p]
    p =[point for point in cube.vertices]
    
    for point in p:
        pygame.draw.circle(screen, BLUE, (point[0], point[1]), 4)

    for edge in cube.edges:
        pygame.draw.line(screen, RED, (p[edge[0]][0], p[edge[0]][1]), (p[edge[1]][0], p[edge[1]][1]), width=1)




    # Update the display
    pygame.display.update()
    clock.tick(fps)

# Quit Pygame
pygame.quit()