%addpath ~/bin/matlab/mesh/
%addpath Intersection

%mesh = mesh_read('~/meshes/MeshsegBenchmark-1.0-full/data/off/364.off');
%mesh = mesh_read('part.off');
%mesh = mesh_normalize(mesh);

filename = '001';
base_mesh = mesh_read(['/home/oliver/shared/Geosemantics/Data3/Lamps/' filename '.off']);
seg = load(['/home/oliver/shared/Geosemantics/Data3/Lamps/' filename '.seg']);

geom = part_geometry(base_mesh, seg);
seg_num = 3;
mesh = struct();
mesh.vertices = geom{seg_num}.vertices;
mesh.faces = geom{seg_num}.faces;
mesh = mesh_normalize(mesh);

tic;
num_samples = 50;
num_bins = 50;
%[vote, pos_theta, pos_phi, pos_d, samp] = sym_voting(mesh, num_samples, num_bins);
%num_bins = 350;
[score, pos_theta, pos_phi, samp] = sym_score_center(mesh, num_samples, num_bins);
%mesh_show_color(mesh);
%hold on;
%plot3(samp(:, 1), samp(:, 2), samp(:, 3), 'b.');
[C, val] = sym_extract_planes(mesh, score, pos_theta, pos_phi, samp);
obb = get_obb(C(1,1:3), C(2,1:3), C(3,1:3), mesh);
toc
[g, f] = check_sym2(obb, mesh, samp);
g
f
figure; plot_obb(mesh, obb);
