import numpy as np
from graph.OBB import OBB_3D
from graph.utils import normalize_vector

class Space_Category():
    state = ['Surround', 'Object_In', 'Intersect', 'Object_Out']
    Direction = ['Top', 'Bottom', 'Front', 'Back', 'Left', 'Right']
    Vertical_Direction = ['right-above', 'above', 'left-above', 
                        'right', 'center', 'left', 
                        'right-below', 'below', 'left-below']
    st_num = len(state)
    dr_num = len(Direction)
    vdr_num = len(Vertical_Direction)
    all_num = st_num + dr_num + vdr_num

class Space_Relationship():
    def __init__(self):  
        self.state = Space_Category.state
        self.Direction = Space_Category.Direction
        self.Vertical_Direction = Space_Category.Vertical_Direction
    
    def get_relation(self, points1, box1, points2, box2):
        iou1 = box2.iou_by_point(points1)
        iou2 = box1.iou_by_point(points2)

        state = None
        if iou2 >= 0.9:
            state = self.state.index('Surround')
        if iou1 >= 0.9:
            state = self.state.index('Object_In')
        if state is None:
            if iou1 <= 0.1:
                state = self.state.index('Object_Out')
            else:
                state = self.state.index('Intersect')
        state = self.state[state]
        
        face_vector = box2.face_center - box1.center
        dis = np.linalg.norm(face_vector, axis=1)
        min_index = np.argmin(dis)
        direct = self.Direction[min_index]
        # dis = np.zeros(6, dtype=float)
        # face_vector = box2.face_center - box1.center
        # # top + bottom
        # dis[:2] = np.abs(np.dot(face_vector[:2], normalize_vector(box2.direction[2])))
        # # front + back
        # dis[2:4] = np.abs(np.dot(face_vector[2:4], normalize_vector(box2.direction[0])))
        # # left + right
        # dis[4:6] = np.abs(np.dot(face_vector[4:6], normalize_vector(box2.direction[1])))
        # min_index = np.argmin(dis)
        # direct = self.Direction[min_index]

        face_vector = -face_vector
        x = np.dot(face_vector[min_index], normalize_vector(box2.direction[0]))
        y = np.dot(face_vector[min_index], normalize_vector(box2.direction[1]))
        z = np.dot(face_vector[min_index], normalize_vector(box2.direction[2]))
        x_index = 0
        y_index = 0
        z_index = 0
        if abs(x) <= (box2.size[0] / 6):
            x_index += 1
        elif abs(x) > (box2.size[0] / 6) and x < 0:
            x_index += 2
        if abs(y) <= (box2.size[1] / 6):
            y_index += 1
        elif abs(y) > (box2.size[1] / 6) and y < 0:
            y_index += 2
        if abs(z) <= (box2.size[2] / 6):
            z_index += 1
        elif abs(z) > (box2.size[2] / 6) and z < 0:
            z_index += 2
        # top + bottom
        if min_index == 0 or min_index == 1:
            # face_point = face_vector[min_index] - z * normalize_vector(box2.direction[2])
            v_diret_index = y_index + x_index * 3
        # front + back
        elif min_index == 2 or min_index == 3:
            v_diret_index = y_index + z_index * 3
        else:
            v_diret_index = x_index + z_index * 3
        v_direct = self.Vertical_Direction[v_diret_index]

        # from mpl_toolkits.mplot3d import axes3d
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.scatter(box1.vertixs[:, 0], box1.vertixs[:, 1], box1.vertixs[:, 2], s=5, c='r', marker='o')
        # # ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], s=5, c='g')
        # ax.scatter(box1.center[0], box1.center[1], box1.center[2], s=20, c='k', marker='x')

        # ax.scatter(box2.vertixs[:, 0], box2.vertixs[:, 1], box2.vertixs[:, 2], s=5, c='r', marker='o')
        # # ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], s=5, c='b')
        # ax.scatter(box2.face_center[min_index][0], box2.face_center[min_index][1], box2.face_center[min_index][2], s=20, c='k', marker='x')

        # axis_x = np.concatenate((box2.axis_points[0], box2.axis_points[1])).reshape(2, 3)
        # ax.plot(axis_x[:, 0], axis_x[:, 1], axis_x[:, 2], c='r')
        # axis_y = np.concatenate((box2.axis_points[0], box2.axis_points[2])).reshape(2, 3)
        # ax.plot(axis_y[:, 0], axis_y[:, 1], axis_y[:, 2], c='g')
        # axis_z = np.concatenate((box2.axis_points[0], box2.axis_points[3])).reshape(2, 3)
        # ax.plot(axis_z[:, 0], axis_z[:, 1], axis_z[:, 2], c='b')
        # plt.show()

        return state, direct, v_direct
            



