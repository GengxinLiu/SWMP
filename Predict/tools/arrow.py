import open3d as o3d
from matplotlib.cm import get_cmap
import imageio
import numpy as np


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/(vec[0] + 1e-5))
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/(vec[2]  + 1e-6))
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)

def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.1
    cylinder_height = scale*0.5
    cone_radius = scale/20
    cylinder_radius = scale/40
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None, color=[1,0,0]):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 0.8
    Ry = Rz = np.eye(3)
    # T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
        scale = np.linalg.norm(vec)/5
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        # scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    mesh.paint_uniform_color(color)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)