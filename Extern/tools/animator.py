import numpy as np
import open3d as o3d
from tools import transform

def create_anim(pc, seg, mov_idx, pos, dir, mo_type, diff, num_frame, num_point=2048):    
	pc_frame = np.zeros(shape=(num_frame + 1, num_point, 3))

	for i in range(num_frame + 1):
		tmp_pc = np.copy(pc)
		frame_diff = i * diff / (num_frame)
		
		if len(seg.shape) > 1:
			seg = seg[0]
		
		frame_mov = tmp_pc[seg == mov_idx] - pos
		if mo_type == 0: # Rotation
			vertice = np.insert(frame_mov, 3, 1, axis=1)
			M = transform.rotation_matrix(np.pi*frame_diff/180, dir)
			vertice = np.matmul(M, vertice.T) # vertice: 3+1, num_select_points
			frame_mov = vertice.T[:, :3]
		if mo_type == 1:
			frame_mov += dir * frame_diff
		if mo_type == 2: # Rotation + Transition
			vertice = np.insert(frame_mov, 3, 1, axis=1)
			M = transform.rotation_matrix(np.pi*frame_diff*30/180, dir)
			vertice = np.matmul(M, vertice.T) # vertice: 3+1, num_select_points
			frame_mov = vertice.T[:, :3]
			frame_mov += dir * frame_diff
		tmp_pc[seg == mov_idx] = frame_mov + pos
		pc_frame[i] = tmp_pc

	return pc_frame