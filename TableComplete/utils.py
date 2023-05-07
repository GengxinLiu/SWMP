import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import random
import open3d
import os


def show_points(coords, colors=None, labels=None, normals=None):
    """
    Draw point clods
        :param coords: [N, 3]
        :param colors: [N, 3]
        :param labels: [N]
    """

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(coords)

    if colors is not None:
        if np.max(colors) > 20:  # 0-255
            colors /= 255.
        pc.colors = o3d.utility.Vector3dVector(colors)
    if labels is not None:
        label_set = np.unique(labels)
        cmap = plt.cm.get_cmap('jet', len(label_set))
        label_colors = np.zeros((len(labels), 3), dtype=float)
        for i, l in enumerate(label_set):
            idx = np.where(labels == l)
            # TODO: change color
            if l == 100 or l == 100:
                label_colors[idx, :] = [0, 0, 0]
            else:
                label_colors[idx, :] = cmap(i)[:3]
                # if l == 24:
                #     label_colors[idx, :] = [255, 255, 255]
        pc.colors = o3d.utility.Vector3dVector(label_colors)
    show_normals = False
    if normals is not None:
        show_normals = True
        pc.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pc], point_show_normal=show_normals, width=500, height=400, window_name='base')


def lossOneOfAll(fname, drawer_front_id, data):
    """
    Determine whether the current drawer_box needs to be completed
    :param shape_pc_ids:
    :param fname:
    :param drawer_front_id: id of the drawer_front, used to locate the corresponding drawer_box
    :param data: j
    :return:
    """
    if fname in ['19571', '19872', '22815', '23791', '24268', '24566', '27189', '29435']:
        return None

    def check(node, drawer_front_id, res):
        if "children" in node:
            children = node["children"]
            for child in children:
                if child["name"] == "drawer_box":
                    num_drawer_side = 0
                    num_drawer_bottom = 0
                    num_drawer_back = 0
                    find = False
                    for drawer_child in child["children"]:
                        if drawer_child["name"] == "drawer_side":
                            num_drawer_side += 1
                        if drawer_child["name"] == "drawer_back":
                            num_drawer_back += 1
                        if drawer_child["name"] == "drawer_bottom":
                            num_drawer_bottom += 1
                        if drawer_child["name"] == "drawer_front" and drawer_child["id"] == drawer_front_id:
                            print(f"find drawer front {drawer_front_id}")
                            find = True
                    if find:
                        # print("Drawer box id {} has {} side {} back  {} bottom".format(child["id"], num_drawer_side,
                        #                                                                num_drawer_back, num_drawer_bottom))
                        res["drawer_box_id"] = child["id"]
                        res["num_side"] = num_drawer_side
                        res["num_back"] = num_drawer_back
                        res["num_bottom"] = num_drawer_bottom
                        return
                check(child, drawer_front_id, res)

    res = {}
    check(data[0], drawer_front_id, res)
    return res["drawer_box_id"], res["num_side"] < 2, res["num_back"] < 1, res["num_bottom"] < 1


def get_vertex(shape_pc, shape_pc_ids, l, fname):
    idx = np.where(shape_pc_ids == l)
    min_x = min(shape_pc[idx][:, 0])
    max_x = max(shape_pc[idx][:, 0])
    min_y = min(shape_pc[idx][:, 1])
    max_y = max(shape_pc[idx][:, 1])
    min_z = min(shape_pc[idx][:, 2])
    max_z = max(shape_pc[idx][:, 2])
    if fname == '19899':
        min_z = 0.255
    if fname == '20056' and l == 19:
        min_x = 0.01
        min_z = 0.32
    if fname == '21739':
        min_z = 0.50
    if fname == '23549':
        min_y = -0.2
        max_y = 0.06
    if fname == '24671':
        min_z = 0.32
    if fname == '25403':
        if l == 39:
            min_x = -0.52
            max_x = 0.65
            min_y = 0
        if l == 42:
            min_x = -0.4
            max_x = 0.4
            min_y = -0.26
    if fname == '25425':
        min_x = -0.45
        max_x = 0.43
        if l == 32:
            min_y = -0.4
    if fname == '26424':
        min_z = 0.31
    if fname == '28203':
        min_x = -0.45
        max_x = 0.53
        min_y = -0.04
        max_y = 0.09
        max_z = -0.22
    if fname == '28251':
        min_x = 0.17
        max_x = 0.61
    if fname == '28555' and l == 32:
        min_x = 0.275
        max_x = 0.623
        max_y = 0.06
        min_z = -0.30443
        max_z = -0.287319
    if fname == '31053' and l == 16:
        min_x = 0.318
        max_y = -0.17
        min_z = 0.361
    if fname == '31271':
        min_z = 0.123
    if fname == '31963':
        min_z = 0.306
    if fname == '31549':
        min_z = 0.267
    if fname == '32664':
        max_y = 0.22
        min_y = 0.04
        min_z = 0.1
    if fname == '33118' and l == 32:
        min_y = 0.2
    if fname == '33155':
        min_y = -0.07
    if fname == '33301' and l == 25:
        min_z = 0.48
    if fname == '33694':
        min_z = 0.46
    if fname == '33915':
        max_z = -0.3
    if fname == '34156' and l == 39:
        max_x = -0.347
    if fname == '34200':
        min_z = 0.2
    if fname == '34236' and l == 35:
        min_x = -0.262

    if max_x - min_x > 0.3 and max_z - min_z > 0.3:
        longFront = True
    else:
        longFront = False
    # print(l,min_x,max_x,min_y,max_y,min_z,max_z)
    dx = (max_x - min_x) * 0.05
    dy = (max_y - min_y) * 0.05
    dz = (max_z - min_z) * 0.05

    dir = 1
    dirbox = []
    for i in range(5):
        dirbox.append(False)
    for point in shape_pc:
        if max_z - min_z < max_x - min_x and point[1] > max_y and point[2] > min_z - 0.3 and point[2] < max_z - 0.2 and \
                point[0] > min_x and point[0] < max_x:
            dirbox[1] = True  # front
            break
    for point in shape_pc:
        if max_z - min_z < max_x - min_x and point[1] > max_y and point[2] > max_z + 0.2 and point[2] < max_z + 0.3 and \
                point[0] > min_x and point[0] < max_x:
            dirbox[2] = True  # back
            break
    for point in shape_pc:
        if max_x - min_x < max_z - min_z and point[1] > max_y and point[0] > max_x - 0.3 and point[0] < max_x - 0.2 and \
                point[2] > min_z and point[2] < max_z:
            dirbox[3] = True  # left
            break
    for point in shape_pc:
        if max_x - min_x < max_z - min_z and point[1] > max_y and point[0] > max_x + 0.2 and point[0] < max_x + 0.3 and \
                point[2] > min_z and point[2] < max_z:
            dirbox[4] = True  # right
            break
    if dirbox[1] or dirbox[2]:
        if dirbox[1] and dirbox[2]:
            N1 = 0
            N2 = 0
            for point in shape_pc:
                if point[2] < min_z:
                    N1 += 1
                if point[2] > max_z:
                    N2 += 1
            if N1 > N2:
                dir = 1
            else:
                dir = 2
        else:
            if dirbox[1]:
                dir = 1
            if dirbox[2]:
                dir = 2
    if dirbox[3] or dirbox[4]:
        if dirbox[3] and dirbox[4]:
            N3 = 0
            N4 = 0
            for point in shape_pc:
                if point[0] < min_x:
                    N3 += 1
                if point[0] > max_x:
                    N4 += 1
            if N3 > N4:
                dir = 3
            else:
                dir = 4
        else:
            if dirbox[3]:
                dir = 3
            if dirbox[4]:
                dir = 4
    if fname == '19898':
        dir = 1
    if fname == '20760' and l == 39:
        dir = 3
    if fname == '26911':
        dir = 3
    if fname == '33817':
        if dir == 2:
            dir = 1
        elif dir == 3:
            dir = 4
        elif dir == 4:
            dir = 3
    print("dir:", dir)
    if dir == 1:
        return [min_x + dx, min_y + dy, min_z], [max_x - dx, max_y - dy, min_z], dir, longFront
    if dir == 2:
        return [min_x + dx, min_y + dy, max_z], [max_x - dx, max_y - dy, max_z], dir, longFront
    if dir == 3:
        return [min_x, min_y + dy, min_z + dz], [min_x, max_y - dy, max_z - dz], dir, longFront
    if dir == 4:
        return [max_x, min_y + dy, min_z + dz], [max_x, max_y - dy, max_z - dz], dir, longFront


def isContain(shape_pc, p1, p2, fname):
    if fname == ['22997', '33380', '34236']:
        return False
    dx = 0.04
    dy = (p2[1] - p1[1]) * 0.35
    dz = 0.04
    for point in shape_pc:
        if point[0] > p1[0] + dx and point[0] < p2[0] - dx and point[1] > p1[1] + dy and point[1] < p2[1] - dy and \
                point[2] > p1[2] + dz and point[2] < p2[2] - dz:
            return True
    return False


def isOutBounds(shape_pc, p1, p2, dir):
    if dir == 1:
        if p1[2] < min(shape_pc[:, 2]):
            return True
        else:
            return False
    if dir == 2:
        if p2[2] > max(shape_pc[:, 2]):
            return True
        else:
            return False
    if dir == 3:
        if p1[0] < min(shape_pc[:, 0]):
            return True
        else:
            return False
    if dir == 4:
        if p2[0] > max(shape_pc[:, 0]):
            return True
        else:
            return False


def judgeCollision(point_set, fname):
    N = len(point_set)
    isCollision = False
    dt = 0.5
    visited = []
    for i in range(N):
        visited.append(False)
    for i in range(0, N):
        for j in range(i + 1, N):
            if not visited[i] and not visited[j]:
                if point_set[i][0] == 1 and point_set[j][0] == 2:
                    if abs((point_set[i][1][0] + point_set[i][2][0]) / 2 - (
                            point_set[j][1][0] + point_set[j][2][0]) / 2) < 0.1:
                        if point_set[i][1][2] < point_set[j][2][2] and point_set[i][2][2] > point_set[j][1][2]:
                            isCollision = True
                            visited[i] = True
                            visited[j] = True
                            point_set[i][1][2] += dt * (point_set[i][2][2] - point_set[i][1][2])
                            point_set[j][2][2] -= dt * (point_set[j][2][2] - point_set[j][1][2])
                if point_set[i][0] == 2 and point_set[j][0] == 1:
                    if abs((point_set[i][1][0] + point_set[i][2][0]) / 2 - (
                            point_set[j][1][0] + point_set[j][2][0]) / 2) < 0.1:
                        if point_set[j][1][2] < point_set[i][2][2] and point_set[j][2][2] > point_set[i][1][2]:
                            isCollision = True
                            visited[i] = True
                            visited[j] = True
                            point_set[j][1][2] += dt * (point_set[j][2][2] - point_set[j][1][2])
                            point_set[i][2][2] -= dt * (point_set[i][2][2] - point_set[i][1][2])
                if point_set[i][0] == 3 and point_set[j][0] == 4:
                    if abs((point_set[i][1][2] + point_set[i][2][2]) / 2 - (
                            point_set[j][1][2] + point_set[j][2][2]) / 2) < 0.1:
                        if point_set[i][1][0] < point_set[j][2][0] and point_set[i][2][0] > point_set[j][1][0]:
                            isCollision = True
                            visited[i] = True
                            visited[j] = True
                            point_set[i][1][0] += dt * (point_set[i][2][0] - point_set[i][1][0])
                            point_set[j][2][0] -= dt * (point_set[j][2][0] - point_set[j][1][0])
                if point_set[i][0] == 4 and point_set[j][0] == 3:
                    if abs((point_set[i][1][2] + point_set[i][2][2]) / 2 - (
                            point_set[j][1][2] + point_set[j][2][2]) / 2) < 0.1:
                        if point_set[j][1][0] < point_set[i][2][0] and point_set[j][2][0] > point_set[i][1][0]:
                            isCollision = True
                            visited[i] = True
                            visited[j] = True
                            point_set[j][1][0] += dt * (point_set[j][2][0] - point_set[j][1][0])
                            point_set[i][2][0] -= dt * (point_set[i][2][0] - point_set[i][1][0])
                            if fname == '21240':
                                point_set[j][1][0] += 0.6 * (point_set[j][2][0] - point_set[j][1][0])
                                point_set[i][2][0] -= 0.6 * (point_set[i][2][0] - point_set[i][1][0])
    if isCollision:
        print("collision")
    return point_set, isCollision


def adjustShape(point_set, fname):
    N = len(point_set)
    k = 1.1
    maxError = 0.1
    resultbox = []
    hasAdjust = False
    if fname == '24276':
        return point_set, hasAdjust
    for i in range(N):
        if point_set[i][0] == 1:
            lenz = point_set[i][2][2] - point_set[i][1][2]
            lenx = point_set[i][2][0] - point_set[i][1][0]
            if lenz > k * lenx:
                hasAdjust = True
                flag = False
                for res in resultbox:
                    print(res)
                    if abs(k * lenx - res) < maxError:
                        print("in res")
                        flag = True
                        point_set[i][1][2] = point_set[i][2][2] - res
                        break
                if not flag:
                    point_set[i][1][2] = point_set[i][2][2] - k * lenx
                    resultbox.append(k * lenx)
        if point_set[i][0] == 2:
            lenz = point_set[i][2][2] - point_set[i][1][2]
            lenx = point_set[i][2][0] - point_set[i][1][0]
            if lenz > k * lenx:
                hasAdjust = True
                flag = False
                for res in resultbox:
                    if abs(k * lenx - res) < maxError:
                        print("in res")
                        flag = True
                        point_set[i][2][2] = point_set[i][1][2] + res
                        break
                if not flag:
                    point_set[i][2][2] = point_set[i][1][2] + k * lenx
                    resultbox.append(k * lenx)
        if point_set[i][0] == 3:
            lenz = point_set[i][2][2] - point_set[i][1][2]
            lenx = point_set[i][2][0] - point_set[i][1][0]
            if lenx > k * lenz:
                hasAdjust = True
                flag = False
                for res in resultbox:
                    if abs(k * lenz - res) < maxError:
                        print("in res")
                        flag = True
                        point_set[i][1][0] = point_set[i][2][0] - res
                        break
                if not flag:
                    point_set[i][1][0] = point_set[i][2][0] - k * lenz
                    resultbox.append(k * lenz)
        if point_set[i][0] == 4:
            lenz = point_set[i][2][2] - point_set[i][1][2]
            lenx = point_set[i][2][0] - point_set[i][1][0]
            if lenx > k * lenz:
                hasAdjust = True
                flag = False
                for res in resultbox:
                    if abs(k * lenz - res) < maxError:
                        print("in res")
                        flag = True
                        point_set[i][2][0] = point_set[i][1][0] + res
                        break
                if not flag:
                    point_set[i][2][0] = point_set[i][1][0] + k * lenz
                    resultbox.append(k * lenz)
    return point_set, hasAdjust


def creatObjPlane(dir, p1_tmp, p2_tmp, mode, fname, i, data_root, obj_dir):
    mesh = open3d.io.read_triangle_mesh('cube.obj')

    vertices = np.asarray(mesh.vertices)
    min_x = min(vertices[:, 0])
    max_x = max(vertices[:, 0])
    min_y = min(vertices[:, 1])
    max_y = max(vertices[:, 1])
    min_z = min(vertices[:, 2])
    max_z = max(vertices[:, 2])
    T = np.array(
        [[(p2_tmp[0] - p1_tmp[0]) / (max_x - min_x), 0, 0, 0], [0, (p2_tmp[1] - p1_tmp[1]) / (max_y - min_y), 0, 0],
         [0, 0, (p2_tmp[2] - p1_tmp[2]) / (max_z - min_z), 0], [0, 0, 0, 1]])
    mesh = mesh.transform(T)
    vertices = np.asarray(mesh.vertices)
    min_x = min(vertices[:, 0])
    min_y = min(vertices[:, 1])
    min_z = min(vertices[:, 2])
    mesh.translate([p1_tmp[0] - min_x, p1_tmp[1] - min_y, p1_tmp[2] - min_z])

    save_root = os.path.join(data_root, fname, obj_dir, str(i))
    if mode == 1:
        open3d.io.write_triangle_mesh(save_root + '_bottom.obj', mesh)
    if mode == 2:
        if dir == 1:
            open3d.io.write_triangle_mesh(save_root + '_leftside.obj', mesh)
        if dir == 2:
            open3d.io.write_triangle_mesh(save_root + '_rightside.obj', mesh)
        if dir == 3:
            open3d.io.write_triangle_mesh(save_root + '_back.obj', mesh)
    if mode == 3:
        if dir == 3:
            open3d.io.write_triangle_mesh(save_root + '_leftside.obj', mesh)
        if dir == 4:
            open3d.io.write_triangle_mesh(save_root + '_rightside.obj', mesh)
        if dir == 2:
            open3d.io.write_triangle_mesh(save_root + '_back.obj', mesh)
    if mode == 4:
        if dir == 2:
            open3d.io.write_triangle_mesh(save_root + '_leftside.obj', mesh)
        if dir == 1:
            open3d.io.write_triangle_mesh(save_root + '_rightside.obj', mesh)
        if dir == 4:
            open3d.io.write_triangle_mesh(save_root + '_back.obj', mesh)
    if mode == 5:
        if dir == 4:
            open3d.io.write_triangle_mesh(save_root + '_leftside.obj', mesh)
        if dir == 3:
            open3d.io.write_triangle_mesh(save_root + '_rightside.obj', mesh)
        if dir == 1:
            open3d.io.write_triangle_mesh(save_root + '_back.obj', mesh)


def getIdSemantic(root, nodeDict):
    nodeDict[str(root["id"])] = root["name"]
    if "children" in root:
        for child in root["children"]:
            getIdSemantic(child, nodeDict)
