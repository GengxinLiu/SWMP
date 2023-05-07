import json
import open3d as o3d
import numpy as np
import os
import trimesh
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('bmh')

default_color = [0,0.5,1]
cube = np.array([
    [0,0,0], [1,0,0], [1,1,0], [0,1,0], 
	[0,0,1], [1,0,1], [1,1,1], [0,1,1], 
])

'''plt figure'''
def plt_show_save(data, title, save_path=None, xname='', bins=50):
	plt.cla()
	plt.figure(figsize=(12,9))
	if type(data) == dict:
		plt.bar(data.keys(), data.values())
		# plt.xticks(rotation=90)
	else:
		plt.hist(data, bins=bins)
	plt.title(title)
	plt.ylabel('value')
	plt.xlabel(xname)
	if save_path is not None:
		plt.savefig(save_path)
	else:
		plt.show()

def get_pcd(pc, color=default_color):
	pc = np.array(pc)
	
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc)
	pcd.paint_uniform_color(color) # 默认是彩虹色过渡，这里指定染色
	return pcd

def get_o3d_FOR(origin=[0, 0, 0],size=0.1):
	
	mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
	mesh_frame.translate(origin)
	return(mesh_frame)

def show_pcds(pcds, wname='Open3D', FOR=0.1):
	
	if FOR:
		pcds.append(get_o3d_FOR(size = FOR))
	o3d.visualization.draw_geometries(pcds, width=800, height=800, window_name=wname)

def csv2box(csv_path):
	
	obb_info = np.loadtxt(open(csv_path, 'r'),delimiter = ",") # (5,4)

	center = obb_info[0,:3] 
	dirs = 0.5 * (obb_info[2:,:3] * obb_info[2:,-1].reshape(3,1) ) 
	val = cube*2 - 1
	vec = np.matmul(val, dirs) # (8,3)@(3,3)
	corner = center.reshape(1,3) + vec
	return corner,dirs

def add_thickness(pc, direction, scale):
	direction = direction / np.linalg.norm(direction)
	noise = np.random.normal(0, scale, (pc.shape[0],1))
	return pc + noise * direction.reshape(1,3)

def PCA(data, sort=True):
    average_data = np.mean(data,axis=0)       
    decentration_matrix = data - average_data   
    H = np.dot(decentration_matrix.T,decentration_matrix)  
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H) 

    if sort:
        sort = eigenvalues.argsort()[::-1]      
        eigenvalues = eigenvalues[sort]        
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

def box_from_pc(pc, color=default_color, aabb=False, return_corner=True):
	pcd = get_pcd(pc)

	box = pcd.get_axis_aligned_bounding_box() if aabb else \
			pcd.get_oriented_bounding_box()
	if return_corner:
		corner = np.array(box.get_box_points())
		return corner
	else:
		box.color = color
		return box

def box_from_corner(corner, color=default_color, aabb=False):
	corner = np.asarray(corner)
	box = o3d.geometry.AxisAlignedBoundingBox() if aabb else \
		o3d.geometry.OrientedBoundingBox()
	box = box.create_from_points(o3d.utility.Vector3dVector(corner))
	box.color = color
	return box

def box2cen_dir(box:np.ndarray):
	centers = np.zeros((6,3))
	sorted_box = sort_pts(box)
	
	v1 = sorted_box[1]-sorted_box[0]
	v2 = sorted_box[3]-sorted_box[0]
	cos = v1@v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
	if abs(cos) < 0.001:
		tmp = sorted_box[3].copy()
		sorted_box[3] =sorted_box[4]
		sorted_box[4] = tmp
	# 0246, 0145
	centers[0] = sorted_box[:4].mean(axis=0)
	centers[1] = sorted_box[[0,2,4,6]].mean(axis=0)
	centers[2] = sorted_box[[0,1,4,5]].mean(axis=0)
	centers[3:] = 2 * box.mean(0).reshape(1,3) - centers[:3]
	return centers

def box2dir(box:np.ndarray):
	sorted_box = np.array(sorted(box, key = lambda x:x[0]) )
	dirs3 = sorted_box[1:4] - sorted_box[0].reshape(1,-1)
	cos = cosine(dirs3, dirs3).flatten()
	idx = np.argmin(cos)
	if cos[idx]<1e-3:
		d1 = idx//3
		d2 = idx%3
		left_dir = np.cross(dirs3[d1], dirs3[d2])
		return np.vstack([dirs3[d1], dirs3[d2], left_dir])
	else:
		return None

def aabb_dirs(pc):
	mins = pc.min(0)
	maxs = pc.max(0)
	dirs = np.eye(3,3) * (maxs-mins).reshape(1,3) / 2
	center = (mins + maxs) / 2
	corners = center.reshape(1,3) + (cube*2-1)@dirs
	return corners, dirs

def obb_2dirs(pc, axis, return_corner=True):
	else_axis = [0,1,2]
	else_axis.pop(axis)
	sub_pc = pc[:,else_axis]

	cov_pts = np.cov(sub_pc, y=None, rowvar=False, bias=True)
	v, vect = np.linalg.eig(cov_pts)
	tvect = vect.T
	rpts = np.dot(sub_pc, np.linalg.inv(tvect))
	mina = np.min(rpts, 0)
	maxa = np.max(rpts, 0)
	diff = (maxa - mina)*0.5

	center = mina + diff
	corners =  center.reshape(-1,2) + np.array([
		[-1,-1], [1,-1], [1,1], [-1,1]
	]) * diff.reshape(-1,2)

	corners = np.dot(corners, tvect) # (4,2)
	axis_pc = pc[:, axis]
	axis_min,axis_max = axis_pc.min(), axis_pc.max()
	cor1 = np.insert(corners, axis, axis_min, axis=1)
	cor2 = np.insert(corners, axis, axis_max, axis=1)
	corners = np.vstack([cor1,cor2])
	center = corners.mean(0)

	dirs = (corners[[1,3,4]] - corners[0].reshape(1,3))/2
	if return_corner:
		return corners, dirs
	else:
		return center, dirs

def obb_adjust(pc:np.ndarray, fix_dir:np.array, ori_dir:np.array):
	'''ori_dir should be [0,0,1] or [0,1,0] or [1,0,0]'''
	axis = np.argmax(ori_dir)

	fix_dir = fix_dir / np.linalg.norm(fix_dir)
	ori_dir = ori_dir / np.linalg.norm(ori_dir)
	cro = np.cross(ori_dir, fix_dir)
	cos = ori_dir@fix_dir
	if abs(cos)>0.99:
		return obb_2dirs(pc, axis, True)

	vx = np.array([
		[0,		-cro[2],	cro[1]], 
		[cro[2],	0,		-cro[0]], 
		[-cro[1],	cro[0],	0	] 
	])

	rot_w = np.eye(3,3) + vx + np.matmul(vx,vx) / (1+cos)
	rot_verse = np.linalg.inv(rot_w)

	rot_pc = np.matmul(pc, rot_verse.T)
	center, dirs = obb_2dirs(rot_pc, axis, False) 
	# dirs[-1][:2] = 0
	# dirs[-1,-1] = rot_pc[:,axis].max() - rot_pc[:,axis].min()
	cen = center.reshape(-1,3)
	dirs = np.matmul(dirs, rot_w.T)
	box = (cube*2 - 1)@dirs + cen@rot_w.T

	return box, dirs

def pts2pts_dis(pts1,pts2):
	diff = pts1.reshape((-1, 1, 3)) - pts2.reshape((1, -1, 3))
	distance = (diff**2).reshape((-1,3)).sum(axis=-1)
	return distance

def sort_pts(box):
	uniques = []
	for i in range(3):
		uni = np.unique(box[:,i]).shape[0]
		uniques.append(uni<8) # and uni//2==0
	if sum(uniques)==0: uniques[0] = True
	sorted_box = np.array(sorted(box, key = lambda x:x[uniques].sum()))
	return sorted_box

def pc2mesh(pts):
	
	pts = np.asarray(pts)
	pcd = get_pcd(pts)
	pcd.estimate_normals()

	distances = pcd.compute_nearest_neighbor_distance()
	avg_dist = np.mean(distances)
	radius = 1.5 * avg_dist

	mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
			pcd, o3d.utility.DoubleVector([radius, radius * 2]), )
	# return np.asarray(mesh.triangles)
	return mesh

def pc_from_mesh(obj_path, npoints):
	mesh = o3d.io.read_triangle_mesh(obj_path)
	pts = mesh.sample_points_uniformly(number_of_points=npoints)
	return np.array(pts.points)

def load_mesh(obj_path):
	return trimesh.load(obj_path, 'obj', force='mesh')

def merge_mesh(meshs):
	merged_mesh = trimesh.util.concatenate(meshs)
	return merged_mesh

def write_mesh(mesh, path, normal=False, color=False):
	o3d.io.write_triangle_mesh(
		path, mesh, write_vertex_normals=normal, write_vertex_colors=color
	)

def gen_meshs(obj_folder, hier_tree, npoints=1024):
	all_node_mesh = {}

	for node in hier_tree:
		id_ = node['id']

		if 'children' in node.keys():
			sub_mesh = gen_meshs(obj_folder, node['children'], npoints)
			all_node_mesh = {**all_node_mesh, **sub_mesh}
			child_mesh = [sub_mesh[me['id']] for me in node['children']]
			node_mesh = merge_mesh(child_mesh)
			all_node_mesh[id_] = node_mesh
		else:
			meshs = []
			for obj_name in node['objs']:
				obj_path = os.path.join(obj_folder, obj_name+'.obj')
				mesh = load_mesh(obj_path)
				meshs.append(mesh)
			if len(meshs)>1:
				meshs = merge_mesh(meshs)
			else:
				meshs = meshs[0]
			
			all_node_mesh[id_] = meshs
	
	return all_node_mesh

def get_leaves(tree, only=None, flatten=False, pop_child=True):
	leaf_parts = []

	for node in tree:
		data = node[only] if only is not None else node
		if 'children' not in node.keys():
			leaf_parts.append(data)
		else:
			node_list = get_leaves(node['children'], only, flatten) # [{...},] with parent+children idx
			leaf_parts.extend(node_list)
			if flatten:
				if only == None:
					data = data.copy()
					if pop_child:data.pop('children')
				leaf_parts.append(data)
		
	return leaf_parts

def hier2graphic(hier_tree, parent_id=-1, depth=0):
	all_nodes = {}
	for node in hier_tree:
		renode = {
			'name': node['name'], 
			'objs': node['objs'] if 'objs' in node.keys() else [], 
			'parent': parent_id, 
			'depth': depth, 
			'box': node['box'] if 'box' in node.keys() else [], 
			'brother':[], 
			'children_id': [], 
			'leaves': get_leaves([node], 'id'), 
		}

		if 'children' in node.keys():
			children_nodes = hier2graphic(node['children'], node['id'], depth+1)
			all_nodes = {**all_nodes, **children_nodes}
			renode['children_id'] = [i['id'] for i in node['children']]
		
		all_nodes[node['id']] = renode
		
		for child in renode['children_id']:
			all_nodes[child]['brother'] = renode['children_id'][:]
			all_nodes[child]['brother'].remove(child)

	return all_nodes

def update_mopara(hash_hier, ids=[0]):
	main_child = ids[:]

	for key in ids:
		if hash_hier[key]['children_id'] != []:
			tree, mochild = update_mopara(hash_hier, hash_hier[key]['children_id'] )
			hash_hier = {**hash_hier, **tree}

			mopara = {'jointData':{}, 'joint':'', 'motype':''}
			node = hash_hier[mochild]
			
			if 'ref' in node.keys() and key!=0:
				mopara['jointData'] = node['jointData']
				mopara['joint'] = node['joint']
				if 'motype' in node.keys(): mopara['motype'] = node['motype']
				
				refs = node['ref'][:]
				for idx,ref in enumerate(refs):
					while(hash_hier[ref]['depth'] > hash_hier[key]['depth']):
						ref = hash_hier[ref]['parent']
					refs[idx] = ref

				mopara['ref'] = list(set(refs))
				for ref in mopara['ref']:
					if ref in main_child and ref != key:
						main_child.remove(key)
						break
			hash_hier[key] = {**hash_hier[key], **mopara}
		
		elif 'ref' in hash_hier[key].keys(): 
			refs = hash_hier[key]['ref']
			for idx,ref in enumerate(refs):
				while(hash_hier[ref]['depth'] > hash_hier[key]['depth']):
					ref = hash_hier[ref]['parent']
				hash_hier[key]['ref'][idx] = ref
			
			hash_hier[key]['ref'] = list(set(hash_hier[key]['ref']))
			for ref in hash_hier[key]['ref']:
				if ref in main_child and ref != key:
					main_child.remove(key)
					break

	return hash_hier, main_child[0]

def gen_graph(hier_tree, mobi):
	'''
	将hierarchy tree转化称graph
	'''
	hash_hier = hier2graphic(hier_tree) 
	for idx,node in enumerate(mobi):
		# mobi[idx]['ids'] = [i['id'] for i in node['parts']]

		mopara = {'jointData':{}, 'joint':'', 'motype':''} 
		if node['jointData'] != {}:
			mopara['jointData'] = node['jointData']
			mopara['joint'] = node['joint']
			if 'motype' in node.keys(): mopara['motype'] = node['motype']

			if node['parent'] != -1 and 'parts' in mobi[node['parent']].keys():
				ref = [j['id'] for j in mobi[node['parent']]['parts']]
				mopara['ref'] = ref
				
		for sub_node in node['parts']:
			sub_id = sub_node['id']
			hash_hier[sub_id] = {**hash_hier[sub_id], **mopara}
	
	graph, _ = update_mopara(hash_hier)
	statics = {}
	
	for key in graph.keys():
		if 'ref' in graph[key].keys():
			refs = graph[key]['ref'][:]
			for ref in refs:
				if graph[key]['parent'] != graph[ref]['parent'] or ref == key:
					graph[key]['ref'].remove(ref)
			if graph[key]['ref'] == []:
				graph[key].pop('ref')
	
	for key in graph.keys():
		node = graph[key]
		graph[key]['edges']  = {
			'children':{}, 
			'space':{}
		}
		for child in graph[key]['children_id']:
			graph[key]['edges']['children'][child] = ''
		brothers = graph[key]['brother'][:]

		if 'ref' in graph[key].keys(): 
			for bro in brothers:
				if bro in graph[key]['ref']:
					graph[key]['edges']['space'][bro] = 'motion' 
				else:
					graph[key]['edges']['space'][bro] = 'none'
			graph[key].pop('ref')
		else: 
			for bro in brothers:
				graph[key]['edges']['space'][bro] = 'none' if 'ref' in graph[bro].keys() else 'fixed'
		
	return graph, statics

def ref_count(graph):
	for key in graph.keys():
		edges = graph[key]['edges']['space']
		refs = [r for r in edges.keys() if edges[r]=='motion']
		graph[key]['refs'] = refs
	
	invalids, child_allref, expect = reduce_ref(graph)
	return invalids, expect

def reduce_ref(graph, node_key='0'):
	ref_child = set()

	all_invalid = 0
	flgs = 0
	for child in graph[node_key]['children_id']:
		child = str(child)
		if graph[child]['refs'] == []:
			ref_child.add(int(child))
			
		if graph[child]['children_id'] != []:
			invalid, flg, expect = reduce_ref(graph, child)
			all_invalid += invalid if flg else invalid-1
			flgs += 1-flg 

	children_allref = False
	if len(ref_child)==0 and graph[node_key]['brother'] == []:
		children_allref = True
	elif len(ref_child) and ref_child == set(graph[node_key]['children_id']) \
		and not flgs:
		children_allref = True
	
	all_invalid += len(ref_child)
	# print('%s invalids:%d'%(node_key, all_invalid))
	return all_invalid, children_allref, (flgs if flgs else 1)

'''direction, angle, pos, ...'''
def cosine(dirs, vec, abs=True):
	vec = vec.reshape(-1,3)
	vec = vec / np.linalg.norm(vec, axis=-1).reshape(-1,1) # (1-n, 3) -> (1-n,) or val

	mul_res = dirs@vec.T
	cos = mul_res / np.linalg.norm(dirs, axis=-1).reshape(-1,1)

	if abs: cos = np.abs(cos)
	return cos

def cross(dirs, vec, abs=False):
	# vec = vec / np.linalg.norm(vec)
	cro = np.cross(dirs, vec)
	cro = cro / np.linalg.norm(cro, axis=-1)
	if abs:
		cro = np.abs(cro)
	return cro

def motion_pos(direction, gt_pos, pos):
	direction= direction / np.linalg.norm(direction)
	cro = np.cross(pos - gt_pos.reshape(1,3), direction)
	dis = np.abs(np.linalg.norm(cro, axis=-1))
	min_idx = np.argmin(dis)
	return min_idx, dis[min_idx]

def read_json(json_path):
	return json.load(open(json_path, 'r'))

def get_boolseg(seg:np.ndarray, mov_idx):
	mov_idx = np.array(mov_idx).reshape((-1,1)) # (n,1), and seg(1,N)
	return ( seg.reshape((1,-1)) == mov_idx ).sum(axis=0) == 1.

if __name__ == '__main__':
	pass