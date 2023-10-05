#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 00:36:05 2023

@author: frantisekdracek
"""

import pygame
import numpy as np
import MarchingSquares as MS

GRID_WIDTH = 600
GRID_HEIGHT = 600
W_RESOLUTION = H_RESOLUTION =600


# Constants
CELL_W_SIZE = GRID_WIDTH//W_RESOLUTION 
CELL_H_SIZE = GRID_HEIGHT//H_RESOLUTION 


WHITE = (255, 255, 255)
GRID_LINE_COLOR = (0, 0, 0)
CONTOUR_LINE_COLOR = (128, 0, 128)
VERTEX_ON_COLOR = (0, 128, 255)
VERTEX_OFF_COLOR = (255, 128, 0)




def metaball(x,y, x0=0.5, y0=0.5, trsh=1):
    x=x/GRID_WIDTH
    y = y/GRID_HEIGHT

    f = 1/(np.sqrt((x-x0)**2+(y-y0)**2))
    
    return int(f<=trsh)


def heart(x,y, x0=0.5, y0=0.75, trsh=0.05):
    x=x/GRID_WIDTH
    y = y/GRID_HEIGHT

    f = (x-x0)**2 + ((y-y0) +np.sqrt(abs((x-x0))))**2

    return int(f<=trsh)

def metaballs(x, y, x0,y0, x1, y1, trsh1, trsh2):

    return metaball(x, y, x0=x0, y0=y0, trsh=trsh1)  + metaball(x, y, x0=x0, y0=y0, trsh=trsh2)
    

grid_instance = MS.Grid(GRID_WIDTH, GRID_HEIGHT, W_RESOLUTION, H_RESOLUTION, heart)



# Initialize Pygame
pygame.init()

# Create a window
screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
pygame.display.set_caption("Square Grid")


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(WHITE)

    # Draw grid lines
    for x in range(0, GRID_WIDTH, CELL_W_SIZE):
        pygame.draw.line(screen, GRID_LINE_COLOR, (x, 0), (x, GRID_HEIGHT))
    for y in range(0, GRID_HEIGHT, CELL_H_SIZE):
        pygame.draw.line(screen, GRID_LINE_COLOR, (0, y), (GRID_WIDTH, y))
        
    for edge in grid_instance.edges_list:
        pygame.draw.line(screen, CONTOUR_LINE_COLOR, (edge[0][0], edge[0][1]), (edge[1][0], edge[1][1]), 3)

    if False:
        for row in grid_instance.vertex_grid:
            for vertex in row:     
                if vertex.value ==1:
                    pygame.draw.circle(screen, VERTEX_ON_COLOR, (vertex.x, vertex.y), 3)
                if vertex.value ==0:
                    pygame.draw.circle(screen, VERTEX_OFF_COLOR, (vertex.x, vertex.y), 3)

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()