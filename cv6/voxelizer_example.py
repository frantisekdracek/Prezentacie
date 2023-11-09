#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:35:15 2023

@author: frantisekdracek
"""

import numpy as np
import obj_loader as obj_loader

def get_triangle_bounding_box(t1, t2, t3):
    x_min = min(t1[0], t2[0], t3[0])
    x_max = max(t1[0], t2[0], t3[0])
    y_min = max(t1[1], t2[1], t3[1])
    y_max = max(t1[1], t2[1], t3[1])
    z_min = max(t1[1], t2[1], t3[1])
    z_max = max(t1[1], t2[1], t3[1])
    
    min_c = np.array([x_min, y_min, z_min])
    max_c = np.array([x_max, y_max, z_max])
    
    return np.array([min_c, max_c])


def get_box_triangle_intersection(
        t1, t2, t3, #triangle coordinates
        u, #box centre vector
        c #box extent #half edge of cube
        ):
    
    v1, v2, v3 = t1-u, t2-u, t3-u #shifted triangle
    
    #axes for projection
    #cube normals
    e1 = np.array([1,0,0]) 
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])
    #triangle edges
    edge1 = v2-v1
    edge2 = v3-v2
    edge3 = v1-v3
    #triangle normals 
    triangle_normal = np.cross(edge1, edge2) 

    axes = np.array([
        e1, e2, e3,
        triangle_normal,
        np.cross(e1, edge1),
        np.cross(e1, edge2),
        np.cross(e1, edge3),
        np.cross(e2, edge1),
        np.cross(e2, edge2),
        np.cross(e2, edge3),
        np.cross(e3, edge1),
        np.cross(e3, edge2),
        np.cross(e3, edge3)
        ])
    
    #normalisation of axes
    new_axes = []
    for axis in axes:
        
        new_axis = axis/np.dot(axis, axis)
        new_axes.append(
            new_axis
        )
    new_axes = np.array(new_axes)
    
    separator = False
    #projection to axis
    for axis in axes:
        intersect = axis_projection(axis, v1, v2, v3, c)
        if intersect==True:
            separator = True
    
    return separator


    

def axis_projection(axis, v1, v2, v3, c):
    
    p1 = np.dot(v1, axis)
    p2 = np.dot(v2, axis)
    p3 = np.dot(v3, axis)
    
    r = (c[0]*abs(axis[0])+c[1]*abs(axis[1])+c[2]*abs(axis[2]))
    
    intersect = max(-max(p1, p2, p3), min(p1, p2, p3)) > r
    return intersect
    

def voxelize(bb_min, bb_max, #min, max of voxelized space
             resolution, #vec describin num of voxels
             triangles # triples of vectors
             ):
     
    voxel_array = np.zeros(shape = (resolution[0], resolution[1], resolution[2]))
    
    voxel_size = (bb_max -bb_min)/resolution
    
    for triangle in triangles:
        triangle_bb  = get_triangle_bounding_box(triangle[0], triangle[1], triangle[2])
        min_bb_voxel = ((triangle_bb[0] - bb_min)/voxel_size).astype(int)
        max_bb_voxel = ((triangle_bb[1] - bb_min)/voxel_size).astype(int)
        
        for i in range(min_bb_voxel[0], max_bb_voxel[0]+1):
            for j in range(min_bb_voxel[1], max_bb_voxel[1]+1):
                for k in range(min_bb_voxel[2], max_bb_voxel[2]+1):
                    #coordinate of voxel centre
                    u = bb_min + np.array([(i+1/2)*voxel_size[0], (j+1/2)*voxel_size[1], (k+1/2)*voxel_size[2]])
                    separated = get_box_triangle_intersection(triangle[0], triangle[1], triangle[2], u, voxel_size/2)
                    if separated == False:
                       voxel_array[i,j,k] = 1 
                       
    return voxel_array


if __name__ == '__main__':       
    object_mesh_dict = obj_loader.get_model_from_obj_file("/Users/frantisekdracek/Docs/Modelovacie-a-renderovacie-techniky/cv6/tea_pot.obj")
    tri_vert = []
    for tri in object_mesh_dict['tris']:
        tri_vert.append( [object_mesh_dict['verts'][tri[0]], object_mesh_dict['verts'][tri[1]], object_mesh_dict['verts'][tri[2]]])
    tri_vert = np.array(tri_vert)
                    
    
        
    
    