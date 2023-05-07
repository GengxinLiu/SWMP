import numpy as np
from graph.utils import vec_cos_theta, normalize_vector
import random

class OBB_3D():
    def __init__(self, points) -> None:
        self.center = np.array([0.0, 0.0, 0.0], dtype=float)
        self.direction = np.zeros((3, 3), dtype=float)
        self.size = np.array([0.0, 0.0, 0.0], dtype=float)
        self.face_center = np.zeros((3, 3), dtype=float)

        for i in range(8):
            points[i] = np.array(points[i])
        self.vertixs = np.array(points)
        self.compute_center()
        self.compute_direction_size()
        self.compute_face_center()

    def __repr__(self):
        return f'center:({self.center[0],self.center[1],self.center[2]}), size({self.size[0],self.size[1],self.size[2]}), direct({self.direction[0],self.direction[1],self.direction[2]})'

    def compute_center(self):
        center = np.array([0.0, 0.0, 0.0], dtype=float)
        for i in range(8):
            center[0] += self.vertixs[i, 0]
            center[1] += self.vertixs[i, 1]
            center[2] += self.vertixs[i, 2]
        self.center = center / 8
        # print(self.center)

    def compute_direction_size(self):
        self.direction = np.zeros((3, 3), dtype=float)

        index_z = np.argsort(self.vertixs[:,2])
        array_z = self.vertixs[index_z]

        four_points = array_z[:4, :]
        remain_point = array_z[4:, :]
        four_points = four_points[np.argsort(four_points[:, 0])]
        two_points = four_points[:2, :]
        remain_point = np.concatenate((remain_point, four_points[2:, :]), axis=0) 
        two_points = two_points[np.argsort(two_points[:, 1])]
        vertix1 = two_points[0].reshape(1, 3)
        remain_point = np.concatenate((remain_point, two_points[1:, :]), axis=0) 

        distance = np.linalg.norm(remain_point - vertix1, axis=1)
        vertix2 = remain_point[np.argsort(distance)[0]].reshape(1, 3)
        vertix3 = remain_point[np.argsort(distance)[1]].reshape(1, 3)

        face1 = np.concatenate((vertix1, vertix2, vertix3), axis=0)
        direction1 = self.get_normal(face1)

        remain_point = remain_point[np.argsort(distance)[2:]]
        last_point, last_index = self.find_face_point(face1, direction1, remain_point)
        remain_point = np.delete(remain_point, last_index, axis=0)
        distance = np.linalg.norm(remain_point - vertix1, axis=1)
        v_point = remain_point[np.argsort(distance)[0]].reshape(1, 3)
        size1 =  np.linalg.norm(v_point - vertix1)
         
        face2 = np.concatenate((vertix1, vertix2, v_point), axis=0)
        direction2 = self.get_normal(face2)
        size2 = np.linalg.norm(vertix3 - vertix1)

        face3 = np.concatenate((vertix1, vertix3, v_point), axis=0)
        size3 = np.linalg.norm(vertix2 - vertix1)
        direction3 = self.get_normal(face3)
        
        size = np.array([size1, size2, size3])
        direction = np.concatenate((direction1, direction2, direction3), axis=0).reshape(3, 3)
        axis_point = np.concatenate((v_point, vertix3, vertix2))

        index = list([0, 1, 2])
        z_index = np.argmax(axis_point[:, 2])
        index.remove(z_index)
        x_index = index[0]
        y_index = index[1]
        if (axis_point[index[0]]-vertix1)[0, 1] > (axis_point[index[1]]-vertix1)[0, 1]:
            x_index = index[1]
            y_index = index[0]
        index = np.array([x_index, y_index, z_index])
        self.direction = direction[index]
        self.size = size[index]
        self.axis_points = np.concatenate((vertix1, axis_point[index]))

    
    def compute_face_center(self):
        self.face_center = np.zeros((6, 3), dtype=float)
        x = normalize_vector(self.axis_points[1] - self.axis_points[0])
        y = normalize_vector(self.axis_points[2] - self.axis_points[0])
        z = normalize_vector(self.axis_points[3] - self.axis_points[0])
        # Top
        self.face_center[0] = self.axis_points[0] + (self.size[0] / 2) * x + (self.size[1] / 2) * y + (self.size[2]) * z
        # Bottom
        self.face_center[1] = self.axis_points[0] + (self.size[0] / 2) * x + (self.size[1] / 2) * y + 0.0
        # front
        self.face_center[2] = self.axis_points[0] + (self.size[0]) * x + (self.size[1] / 2) * y + (self.size[2] / 2) * z
        # back
        self.face_center[3] = self.axis_points[0] + 0.0 + (self.size[1] / 2) * y + (self.size[2] / 2) * z
        # left
        self.face_center[4] = self.axis_points[0] + (self.size[0] / 2) * x + 0.0 + (self.size[2] / 2) * z
        # Right
        self.face_center[5] = self.axis_points[0] + (self.size[0] / 2) * x + (self.size[1]) * y + (self.size[2] / 2) * z


    def get_normal(self, points):
        line1 = points[1] - points[0]
        line2 = points[2] - points[0]
        normal = np.cross(line1, line2)
        return normalize_vector(normal)
    

    def find_face_point(self, face, normal, points):
        # print(normal)
        temp_point = np.zeros((3, 3))
        temp_point[1] = face[1]
        temp_point[2] = face[2]
        flag = False
        for i in range(points.shape[0]):
            temp_point[0] = points[i]
            curr_normal = self.get_normal(temp_point)
            theta = round(abs(vec_cos_theta(curr_normal, normal)), 4)
            if theta == 1.0:
                flag = True
                return points[i].reshape(1, 3), i
        if flag == False:
            print("error extract face point")


    def sample_points_random(self, num):
        points = np.repeat(self.axis_points[0].reshape(1, 3), num, axis=0)
        x_points = np.random.uniform(0, np.linalg.norm(self.axis_points[1] - self.axis_points[0]), num).reshape(num, 1)
        y_points = np.random.uniform(0, np.linalg.norm(self.axis_points[2] - self.axis_points[0]), num).reshape(num, 1)
        z_points = np.random.uniform(0, np.linalg.norm(self.axis_points[3] - self.axis_points[0]), num).reshape(num, 1)
        points += np.matmul(x_points, normalize_vector(self.axis_points[1] - self.axis_points[0]).reshape(1, 3))
        points += np.matmul(y_points, normalize_vector(self.axis_points[2] - self.axis_points[0]).reshape(1, 3))
        points += np.matmul(z_points, normalize_vector(self.axis_points[3] - self.axis_points[0]).reshape(1, 3))

        return points
    
    def iou_by_point(self, points):
        points = points - self.center

        x = np.abs(np.dot(points, normalize_vector(self.direction[0])))
        index = (x > self.size[0] / 2)
        x[x <= self.size[0] / 2] = 1
        x[index] = 0
        y = np.abs(np.dot(points, normalize_vector(self.direction[1])))
        index = (y > self.size[0] / 2)
        y[y <= self.size[1] / 2] = 1
        y[index] = 0
        z = np.abs(np.dot(points, normalize_vector(self.direction[2])))
        index = (z > self.size[0] / 2)
        z[z <= self.size[2] / 2] = 1
        z[index] = 0

        inside = x * y * z
        num = np.sum(inside == 1)
        return num / points.shape[0]

