import numpy as np
import json
import os
from configs import DATA_ROOT_DICT
from utils import lossOneOfAll, get_vertex, isContain, isOutBounds, judgeCollision, \
    adjustShape, getIdSemantic, creatObjPlane
import argparse


def createDrawer(shape_pc, shape_pc_ids, fname, data, data_result, data_root, obj_dir, update_mobility=False):
    label_set = np.unique(shape_pc_ids)
    hasDrawer = False
    point_set = []
    global addIdStart
    addIdStart = 1000
    if update_mobility:
        mobility = json.load(open(f'{data_root}/{fname}/mobility_v2.json', 'r'))
    for l in label_set:
        if data[str(l)] == "drawer_front":
            hasDrawer = True
            p1, p2, dir, longFront = get_vertex(shape_pc, shape_pc_ids, l, fname)
            if longFront:
                continue
            if fname == '21990' and l == 26:
                continue
            if fname == '23050' and l == 55:
                continue
            if fname == '23273' and l == 39:
                continue
            if fname == '23508' and (l == 84 or l == 86):
                continue
            if fname == '26614' and l == 17:
                continue
            if fname == '27285' and l == 25:
                continue
            if fname == '34152' and not (l == 54 or l == 56 or l == 58):
                continue
            if fname in ['22136', '22500', '22692', '23911', '24142', '26139', '26553', '26608', '28874', '29152',
                         '31917', '34014', '34235', '34336']:
                continue
            if dir == 1:
                p1[2] -= 0.05
                p2[2] -= 0.05
                while (not isContain(shape_pc, p1, p2, fname) and not isOutBounds(shape_pc, p1, p2, dir)):
                    p1[2] -= 0.01
                p1[2] += 0.1
                p2[2] += 0.05
            if dir == 2:
                p1[2] += 0.05
                p2[2] += 0.05
                while (not isContain(shape_pc, p1, p2, fname) and not isOutBounds(shape_pc, p1, p2, dir)):
                    p2[2] += 0.01
                p1[2] -= 0.05
                p2[2] -= 0.1
            if dir == 3:
                p1[0] -= 0.05
                p2[0] -= 0.05
                while (not isContain(shape_pc, p1, p2, fname) and not isOutBounds(shape_pc, p1, p2, dir)):
                    p1[0] -= 0.01
                p1[0] += 0.1
                p2[0] += 0.05
            if dir == 4:
                p1[0] += 0.05
                p2[0] += 0.05
                while (not isContain(shape_pc, p1, p2, fname) and not isOutBounds(shape_pc, p1, p2, dir)):
                    p2[0] += 0.01
                p1[0] -= 0.05
                p2[0] -= 0.1
            # if lossOneOfAll(shape_pc_ids, fname, l, data) and p2[0] - p1[0] > 0.08 and p2[2] - p1[2] > 0.08:
            res = lossOneOfAll(fname, l, data_result)
            if res is not None:
                drawer_box_id, isSide, isBack, isBottom = res
                if isSide or isBack or isBottom:
                    # print(f"add drawer of {fname}/{drawer_box_id}")
                    point_set.append([dir, p1, p2])

                    point_set, isCollision = judgeCollision(point_set, fname)
                    point_set, hasAdjust = adjustShape(point_set, fname)

                    # mesh_drawer_box = createObj(point_set, fname, drawer_box_id)
                    dir = point_set[0][0]
                    p1 = point_set[0][1]
                    p2 = point_set[0][2]

                    p1_tmp = p1[:]
                    p2_tmp = p2[:]
                    p2_tmp[1] = p1_tmp[1] + 0.002
                    creatObjPlane(dir, p1_tmp, p2_tmp, 1, fname, drawer_box_id, data_root, obj_dir)

                    p1_tmp = p1[:]
                    p2_tmp = p2[:]
                    p2_tmp[0] = p1_tmp[0] + 0.002
                    creatObjPlane(dir, p1_tmp, p2_tmp, 2, fname, drawer_box_id, data_root, obj_dir)

                    p1_tmp = p1[:]
                    p2_tmp = p2[:]
                    p1_tmp[2] = p2_tmp[2] - 0.002
                    creatObjPlane(dir, p1_tmp, p2_tmp, 3, fname, drawer_box_id, data_root, obj_dir)

                    p1_tmp = p1[:]
                    p2_tmp = p2[:]
                    p1_tmp[0] = p2_tmp[0] - 0.002
                    creatObjPlane(dir, p1_tmp, p2_tmp, 4, fname, drawer_box_id, data_root, obj_dir)

                    p1_tmp = p1[:]
                    p2_tmp = p2[:]
                    p2_tmp[2] = p1_tmp[2] + 0.02
                    creatObjPlane(dir, p1_tmp, p2_tmp, 5, fname, drawer_box_id, data_root, obj_dir)

                    if update_mobility:
                        index = -1
                        for find_index in range(len(mobility)):
                            parts = mobility[find_index]["parts"]
                            if index < 0:
                                for subsub in parts:
                                    if int(subsub["id"]) == int(l):
                                        index = find_index
                                        break
                        if index == -1:
                            raise ValueError(f"Can not find id {l} in mobility data!")


                    def updateResult(node):
                        global addIdStart
                        if node["id"] == drawer_box_id:
                            # print("find", drawer_box_id)
                            numBack, numBottom, numSide = 0, 0, 0
                            for drawerChild in node["children"]:
                                if drawerChild["name"] == "drawer_back":
                                    numBack += 1
                                elif drawerChild["name"] == "drawer_side":
                                    numSide += 1
                                elif drawerChild["name"] == "drawer_bottom":
                                    numBottom += 1
                                elif drawerChild["name"] == "drawer_front":
                                    continue
                                else:
                                    f = open("compete_log.txt", "a+")
                                    log = f"{fname} {drawer_box_id} Unknown drawerChild " + drawerChild["name"]
                                    f.write(log + "\n")
                                    print(log)
                                    f.close()
                                    # raise ValueError("Unknown drawerChild")
                            if numBack < 1:
                                # add drawer back
                                node["children"].append({
                                    "text": "Drawer back",
                                    "objs": [
                                        f"{drawer_box_id}_back"
                                    ],
                                    "id": addIdStart,
                                    "name": "drawer_back"
                                })

                                if update_mobility:
                                    mobility[index]["parts"].append({
                                        "id": addIdStart,
                                        "name": "drawer_back",
                                        "children": []
                                    })

                                addIdStart += 1
                            else:
                                os.remove(f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_back.obj")
                                if os.path.exists(f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_back.mtl"):
                                    os.remove(
                                        f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_back.mtl")
                            if numBottom < 1:
                                node["children"].append({
                                    "text": "Drawer bottom",
                                    "objs": [
                                        f"{drawer_box_id}_bottom"
                                    ],
                                    "id": addIdStart,
                                    "name": "drawer_bottom"
                                })
                                if update_mobility:
                                    mobility[index]["parts"].append({
                                        "id": addIdStart,
                                        "name": "drawer_bottom",
                                        "children": []
                                    })
                                addIdStart += 1
                            else:
                                os.remove(
                                    f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_bottom.obj")
                                if os.path.exists(f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_bottom.mtl"):
                                    os.remove(
                                        f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_bottom.mtl")
                            if numSide < 2:
                                if numSide == 1:
                                    f = open("complete_log.txt", "a+")
                                    log = f"{fname} drawer {drawer_box_id} has 1 drawer side, cause ambiguity!!"
                                    f.write(log + "\n")
                                    print(log)
                                    f.close()
                                else:
                                    node["children"].append({
                                        "text": "Drawer side",
                                        "objs": [
                                            f"{drawer_box_id}_leftside"
                                        ],
                                        "id": addIdStart,
                                        "name": "drawer_side"
                                    })
                                    if update_mobility:
                                        mobility[index]["parts"].append({
                                            "id": addIdStart,
                                            "name": "drawer_side",
                                            "children": []
                                        })
                                    addIdStart += 1
                                    node["children"].append({
                                        "text": "Drawer side",
                                        "objs": [
                                            f"{drawer_box_id}_rightside"
                                        ],
                                        "id": addIdStart,
                                        "name": "drawer_side"
                                    })
                                    if update_mobility:
                                        mobility[index]["parts"].append({
                                            "id": addIdStart,
                                            "name": "drawer_side",
                                            "children": []
                                        })
                                    addIdStart += 1
                            else:
                                os.remove(
                                    f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_rightside.obj")
                                os.remove(
                                    f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_leftside.obj")
                                if os.path.exists(f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_rightside.mtl"):
                                    os.remove(
                                        f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_rightside.mtl")
                                if os.path.exists(f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_leftside.mtl"):
                                    os.remove(
                                        f"{data_root}/{fname}/{obj_dir}/{drawer_box_id}_leftside.mtl")
                            f = open("compete_log.txt", "a+")
                            log = fname + "/" + str(drawer_box_id) + " has {} botttom {} side {} back".format(numBottom,
                                                                                                              numSide,
                                                                                                              numBack)
                            f.write(log + "\n")
                            f.close()
                            print(log)
                        else:
                            if "children" in node:
                                for child in node["children"]:
                                    updateResult(child)

                    updateResult(data_result[0])

                    point_set = []
        json.dump(data_result, open(f'{data_root}/{fname}/result_complete.json', 'w'))
        if update_mobility:
            json.dump(mobility, open(f'{data_root}/{fname}/mobility_v2_complete.json', 'w'))

    return hasDrawer, point_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="partnet_mobility", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    if args.dataset == "partnet_mobility":
        data_root = os.path.join(DATA_ROOT_DICT[args.dataset], "dataset")
        fnames = json.load(open(r"model.json", "r"))["table"]
        obj_dir = "textured_objs"
        update_mobility = True
    else:
        data_root = os.path.join(DATA_ROOT_DICT[args.dataset], "dataset", "Table", "data_v0")
        fnames = os.listdir(data_root)
        obj_dir = "objs"
        update_mobility = False

    for fname in fnames:
        result_path = os.path.join(data_root, fname, "result.json")
        if not os.path.exists(result_path):
            print(result_path, "not exist, skip")
            continue
        f = open(result_path, 'r')
        data_result = json.load(f)
        data = {}
        getIdSemantic(data_result[0], data)
        shape_pc = np.loadtxt(os.path.join(data_root, fname, "point_sample", "pts-10000.txt"))
        shape_pc_ids = np.loadtxt(os.path.join(data_root, fname, "point_sample", "label-10000.txt"), dtype=int)
        has_drawer, pointset = createDrawer(shape_pc, shape_pc_ids, fname, data,
                                            data_result, data_root, obj_dir, update_mobility)
        print(fname)
