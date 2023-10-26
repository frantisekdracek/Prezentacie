#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:12:27 2023

@author: frantisekdracek
"""

import pygame
import numpy as np



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



def projection_matrix_simple_perspective(n):
    P = np.array([
        [n, 0, 0, 0],
        [0, n, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0]
        ])
    return P

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

    
def scaling(s_x,
            s_y,
            t_x, #right
            t_y  #left
            ):
    S = np.array([
        [s_x, 0, 0, t_x],
        [0, s_y, 0, t_y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])
    return S


    



class Camera:
    
    def __init__(self):
        self._position = np.array([0, 0, 0, 0])
        self._rotation = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        #self._field_of_view_axis = np.array([0,0,1)
        self._near = 0.3
        self._far = 1000
        self._right = 1
        self._left = -1
        self._top = 1
        self._bottom = -1 
        self._projection = self.camera_perspective_projection_matrix(
            self._position, 
            self._rotation,
            self._right,
            self._left,
            self._top,
            self._bottom,
            self._near,
            self._far 
            )
        
    
    def update_projection(self):
        self._projection = self.camera_perspective_projection_matrix(
            self._position, 
            self._rotation,
            self._right,
            self._left,
            self._top,
            self._bottom,
            self._near,
            self._far 
            )
        
        return self
        
    
    def set_position(self, position):
        if position.shape[-1] == 3:
            position = np.concatenate((position, np.array([1])))   
        
        self._position = position
            
        self.update_projection()
        
        return self
        
    def set_rotation(self, rotation):
        #todo check dimensions
        
        self._rotation = rotation

        self.update_projection()
        
        return self
    
    def set_fulcrum(self, near, far, right, left, top, bottom):
        self._near = near
        self._far = far
        self._right = right
        self._left = left
        self._top = top
        self._bottom = bottom
        
        self.update_projection()
        
        return self        
            

            
        
        
        
    def camera_perspective_projection_matrix(self,
            c, # vector
            R, # rotation
            r,
            l,
            t,
            b, 
            n,
            f
            ):
    
        #translate camera origin
        T = translation(-c)

        #rotation
        R = R
    
    
        #perspective projection with near and far culling
        P = projection_matrix_simple_perspective_with_culling(n, f)
    
        # fulcrum scaling
        FS = scaling(2/(r-l), 2/(t-b), (r+l)/(r-l), (t+b)/(t-b))
        
        CM = FS@P@R@T
        
        return CM


    def project(self, a):
        if a.shape[-1] == 3:
            a = np.concatenate((a, np.array([1])))
        
        a_projected = self._projection @ a
        a_homegenous = a_projected/a_projected[3]
        
        return a_homegenous

    #todo options for transforming 
    




        
class SceneGraph:
    def __init__(self, mesh = None):
        #todo transforms
        self.mesh = mesh
        #self.set_mesh(mesh)
        
    def set_mesh(self, mesh):
        self.mesh = mesh()
        
        return self
    

        
class Mesh:
    def __init__(self, vertices = None, primitives = None):
        self.vertices = None
        self.z_buffer = None
        self.primitives = None

        self.set_vertices(vertices)
        self.set_primitives(primitives)
    
    def set_vertices(self, vertices):
        if vertices is not None:
            vertices = np.array(vertices)
        self.vertices = vertices
        
        
        return self
    
    def set_primitives(self, primitives):
        if primitives is not None:
            primitives = np.array(primitives)
        #todo check primitive size
        self.primitives = primitives
        
        self.set_z_buffer()
        
        return self
    
    def add_primitive(self, primitive):
        #todo check primitive size
        self.primitves = np.append(self.primitves, primitive)
        self.set_z_buffer()

        return self
    
    def set_z_buffer(self):
        self.z_buffer = np.zeros(self.primitives.shape[0])
        for i in range(self.primitives.shape[0]):
            primitive=self.primitives[i]
            self.z_buffer[i] = 3/(1/self.vertices[primitive[0]][2] + 1/self.vertices[primitive[1]][2] + 1/self.vertices[primitive[2]][2])
        
        sorted_indices = np.argsort(self.z_buffer[::-1])
        self.z_buffer = self.z_buffer[sorted_indices]
        self.primitives = self.primitives[sorted_indices]
        
        return self
        
    def rotate(self, R):
        self.vertices = (R@(self.vertices.T)).T
    
    #todo normals
    #todo z depth
        
    



class Application:
    
    def __init__(self):
        #resolution
        self.WIDTH = 600
        self.HEIGHT = 600

        #colors
        self.WHITE = (255, 255, 255)
        self.BLUE = (20, 120, 240)
        self.RED = (240, 120, 20)
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Display")
        self.clock = pygame.time.Clock()
        self.fps = 10
        self.running = False
        self.scene_graph = None
        self.camera = Camera()
        
    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                #sys.exit()
        
    def render(self):
        self.screen.fill(self.WHITE)        
        v_proj = np.array([self.camera.project(vertex) for vertex in self.scene_graph.mesh.vertices])
        v_proj = [scaling(self.WIDTH/2,self.HEIGHT/2, self.WIDTH/2, self.HEIGHT/2)@point for point in v_proj]

        for point in v_proj:
            print(point)
            pygame.draw.circle(self.screen, self.BLUE, (point[0], point[1]), 4)
            
            
        
        for prim in self.scene_graph.mesh.primitives:
            p0 = v_proj[prim[0]]
            p1 = v_proj[prim[1]]
            p2 = v_proj[prim[2]]
            pygame.draw.polygon(self.screen, self.RED, [[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]]] )

        pygame.display.update()

    
    def run(self):
        self.running = True
        while self.running:
            self.check_events()
            self.render()
            self.clock.tick(self.fps)
            
            
            
if __name__ == "__main__":
    app = Application()
    
    cube = Mesh(
        vertices=np.array([
                 [-1,  1,  1],
                 [ 1,  1,  1],
                 [ 1, -1,  1],
                 [-1, -1,  1],
                 [-1,  1, -1],
                 [ 1,  1, -1],
                 [ 1, -1, -1],
                 [-1, -1, -1]           
                ]),
        primitives=np.array([
            [0, 1, 2], [2, 3, 1], 
            [4, 5, 6], [6, 7, 4],
            [0, 1, 5], [5, 4, 0],
            [3, 2, 6], [6, 7, 3],
            [0, 4, 7], [7, 3, 0],
            [1, 5, 6], [6, 2, 1]
        ])
        )
    cube.rotate(R_y(np.pi/30, homogenous=False))
    
    scene_graph = SceneGraph(mesh=cube)
    
    #set camera
    camera_position = np.array([0,0,-5])
    app.camera.set_position(camera_position)
    app.scene_graph = scene_graph
    
    #create scene graph
    


    app.run()

    
    



"""
    # Clear the screen
    
    #R = R@ R_y(np.pi/100, homogenous = False)#@ R_(np.pi/100
    #camera.set_position(R@camera_position)
    #camera.set_rotation(rotation_from_vectors(-R@camera_position, np.array([0,-1,0])))

    #R = R@ R_y(np.pi/300, homogenous = True)#@ R_(np.pi/100
    R = R_y(np.pi/300*128)
    camera.set_rotation(R)
    
    p = [camera.project(point) for point in cube.vertices]
    p = [scaling(100,100, WIDTH/2, HEIGHT/2)@point for point in p]
    
    for point in p:
        pygame.draw.circle(screen, BLUE, (point[0], point[1]), 3)

    for edge in cube.edges:
        pygame.draw.line(screen, RED, (p[edge[0]][0], p[edge[0]][1]), (p[edge[1]][0], p[edge[1]][1]), width=1)
"""