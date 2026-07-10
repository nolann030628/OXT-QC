import open3d as o3d
import numpy as np



def create_sphere(center, radius=0.025, color=[1, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color) 
    return sphere


def create_coordinate(origin, orientation, size=0.1):
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    # coordinate.rotate(orientation, center=-size/2*np.ones(3))
    coordinate.rotate(orientation, center=np.zeros(3))
    coordinate.translate(origin)
    return coordinate