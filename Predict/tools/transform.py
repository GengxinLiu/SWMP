import numpy as np
import math
import os

def rotation_matrix(angle, direction, point = None):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                      [direction[2], 0.0, -direction[0]],
                      [-direction[1], direction[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def unit_vector(data, axis = None, out = None):
    #Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    if out is None:
        data = np.array(data, dtype = np.float64, copy = True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy = False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def norm_obj(vertices):
    vertices = np.asarray(vertices)
    centroid = np.zeros((3))
    centroid[0] = (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) / 2
    centroid[1] = (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) / 2
    centroid[2] = (np.max(vertices[:, 2]) + np.min(vertices[:, 2])) / 2

    scale = np.zeros((3))
    scale_x = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    scale_y = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    scale_z = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
    scale = np.max([scale_x, scale_y, scale_z])

    return centroid,scale

def add_offset(line, faces_offset):
    plus_faces_list = []
    assert line[0] == 'f'
    str1 = line[1:].split()
    str2 = '/'
    for x in str1:
        m = int(x[:x.index(str2)]) + faces_offset
        plus_faces_list.append(m)
    for j in range(0, len(plus_faces_list), 3):
        face = plus_faces_list[j:j + 3]
    return face

def trans(output_group_dir,filenames,motion,centroid,scale,m,num):
    OFFSET = np.array([0.5, 0.5, 0.5])
    new_faces_list = []
    vert = []
    faces_offset = 0
    with open(os.path.join(output_group_dir, m, m + '_' + str(num) + '.obj'), 'w') as fg:
        for filename in filenames:
            with open(filename, 'rt') as f:
                vertices_num = 0
                print('filename', filename)
                for line in f.readlines():
                    if line[:2] == "v ":
                        vertex2 = (float(line.split(' ')[-3]), float(line.split(' ')[-2]), float(line.split(' ')[-1]))
                        vertex_norm2 = (vertex2 - centroid) / scale + OFFSET
                        vert.append(vertex_norm2)
                        vertices_num = vertices_num + 1
                    if line[:2] == 'f ':
                        new_line = add_offset(line, faces_offset)
                        new_line_str = ' '.join(str(e) for e in new_line)
                        str4 = 'f'
                        new_face_str = str4 + ' ' + new_line_str + '\n'
                        new_faces_list.append(new_face_str)
            faces_offset += vertices_num
        ver = np.asarray(vert)
        r_vertex_list = []
        if motion['type'] == 'R':
            print('r')
            ver = ver - motion['axispos']
            ones_row = np.ones(len(ver), int)
            ver4 = np.insert(ver, 3, ones_row, axis=1)
            M1 = rotation_matrix(math.pi / 180 * motion['angrange']['a'], motion['axisdir'])
            ver_r = np.matmul(M1, ver4.T)
            new_r_ver = ver_r.T[:, 0:3]
            new_r_ver = new_r_ver + motion['axispos']
            for vertex_r in new_r_ver:
                fg.writelines(
                    "v " + str('%.6f' % vertex_r[0]) + " " + str('%.6f' % vertex_r[1]) + " " + str('%.6f' % vertex_r[2]) + "\n")
            fg.writelines('o' + ' ' + m + '_' + str(num) + '\n')
            fg.writelines('g' + ' ' + m + '_' + str(num) + '\n')

        t_vertex_list = []
        if motion['type'] == 'T':
            print('t')
            t = []
            for i in motion['axisdir']:
                t_offset = i * motion['angrange']['a']
                t.append(t_offset)

            t_ver = ver + t
            for vertex_t in t_ver:
                fg.writelines(
                    "v " + str('%.6f' % vertex_t[0]) + " " + str('%.6f' % vertex_t[1]) + " " + str('%.6f' % vertex_t[2]) + "\n")
            fg.writelines('o' + ' ' + m + '_' + str(num) + '\n')
            fg.writelines('g' + ' ' + m + '_' + str(num) + '\n')

        oral_vertex_list = []
        if motion['type'] != 'T' and motion['type'] != 'R':
            for vertex_0 in ver:
                fg.writelines(
                    "v " + str('%.6f' % vertex_0[0]) + " " + str('%.6f' % vertex_0[1]) + " " + str('%.6f' % vertex_0[2]) + "\n")
            fg.writelines('o' + ' ' + m + '_' + str(num) + '\n')
            fg.writelines('g' + ' ' + m + '_' + str(num) + '\n')

        for f in new_faces_list:
            fg.writelines(str(f))