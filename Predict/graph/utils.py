import numpy as np

def vec_cos_theta(vec_a, vec_b):
    """
    Args:
        vec_a: list/np.array [3]
        vec_b: list/np.array [3]
    Returns:
        cos_theta: float
    """
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    len_a = np.linalg.norm(vec_a)
    len_b = np.linalg.norm(vec_b)
    return (np.dot(vec_a, vec_b) / ( len_a * len_b ))

def normalize_vector(vector_a):
    return vector_a / np.linalg.norm(vector_a)