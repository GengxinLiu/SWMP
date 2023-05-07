import torch
# import numpy as np

DEVICE = 'cuda'

def cosine(dirs, vec, abs=True):
	vec = vec.reshape((-1,3))
	vec = vec / torch.norm(vec, dim=-1).reshape(-1,1) # (1-n, 3) -> (1-n,) or val

	mul_res = dirs@vec.T
	cos = mul_res / torch.norm(dirs, dim=-1).reshape(-1,1)

	if abs: cos = torch.abs(cos)
	return cos

def dis_pts2mesh(
	pts, 
	dirs, 
	mesh_pts, 
	mesh_vertices, 
	min_thres, 
	collide=True, 
	return_all=False
):
	thres = 1e3
	pts = torch.as_tensor(pts, device=DEVICE)
	dirs = torch.as_tensor(dirs, device=DEVICE)
	mesh_pts = torch.as_tensor(mesh_pts, device=DEVICE)
	all_dis = torch.as_tensor([thres]*len(dirs), device=DEVICE) # [1e3, ...]

	surf_pts = mesh_pts[mesh_vertices]
	v1 = surf_pts[:, 0] - surf_pts[:, 1]
	v2 = surf_pts[:, 1] - surf_pts[:, 2]
	norm = torch.cross(v1,v2) # norm.shape = (n,3)
	for point in pts: 
		for j, dr in enumerate(dirs): 
			cos = cosine(norm, dr).view(-1)
			index = cos > 0

			norm1 = norm[index] 
			surf = surf_pts[index] 
			vecs = surf - point.reshape(1,1,3) 
			vecs = vecs.reshape(-1, 3) # (m*3, 3)
			cos = cosine(vecs, dr, abs=False).reshape(-1, 3) # (m*3, 1) -> (m, 3)

			if 3 in (cos==0).sum(dim=-1): 
				all_dis[j] = 0
				continue
			index = (cos>0).sum(dim=-1) >= 2 
			surf = surf[index]
			norm1 = norm1[index]
			cos = cos[index]
			vecs = vecs.reshape(-1,3,3)[index]
			mesh_num = vecs.shape[0]
			try:
				sel_idx = torch.argmax(1*(cos > 0), dim=-1) 
			except:
				continue
			sel_vec = torch.zeros((mesh_num, 3), device=DEVICE)
			for i in range(mesh_num):
				sel_vec[i] = vecs[i][sel_idx[i]] 
			
			dis = torch.abs((sel_vec * norm1).sum(dim=-1)) / torch.norm(norm1, dim=-1)
			cos = cosine(norm1, dr).flatten()
			bias = dis / cos 
			ir_points = point.view(1,3) + bias.view(-1,1) * dr.view(1,3)
			vecs = surf - ir_points.reshape(-1,1,3)
			
			c1 = torch.cross(vecs[:, 0], vecs[:, 1]) # (n, 3)
			c2 = torch.cross(vecs[:, 1], vecs[:, 2])
			c3 = torch.cross(vecs[:, 2], vecs[:, 0])
			f1,f2,f3 = (c1*c2).sum(axis=-1), (c2*c3).sum(axis=-1), (c3*c1).sum(axis=1)
			flg = (f1>0) * (f2>0) * (f3>0) 
			
			if flg.sum()>0 or not collide:
				bias = bias[flg]
				select_dis = bias>=min_thres
				if select_dis.sum()<1:continue 
				all_dis[j] = min(all_dis[j], bias[select_dis].min().item())
	
	index = all_dis < thres
	if return_all:
		all_dis[index == False] = 0
		return all_dis.cpu().detach().numpy()
	else:
		avg_dis = all_dis[index].mean() if index.sum()>0 else -1
		return avg_dis

if __name__ == '__main__':
	cube = torch.tensor([
		[0,0,0], [1,0,0], [0,1,0], [0,0,1], 
		[1,1,0], [1,0,1], [0,1,1], [1,1,1], 
	], device=DEVICE, dtype=torch.float32)

	data = {
		'pts':torch.tensor([[0.5, 0.5, 0.5,]], device=DEVICE, dtype=torch.float32), 
		'dirs':torch.tensor([[0, 0, 1]], device=DEVICE, dtype=torch.float32), 
		'mesh_pts':cube, 
		'mesh_vertices':torch.tensor([[1,2,3], [4,5,6], [0,1,3], [0,1,2], [3,6,7]], device=DEVICE)
	}
	dis = dis_pts2mesh(**data)
	print(dis)