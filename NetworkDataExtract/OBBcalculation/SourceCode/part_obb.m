
function bbox = part_obb(mesh)

thresh = 1e-4;

faces = mesh.faces;
points = mesh.vertices;

nsamples = 100;
[~, area] = mesh_face_normals(mesh);
samples = mesh_sample(mesh, area, nsamples);


coh = convhull(points(:,1), points(:,2), points(:,3));
chp = unique(coh);

tic;
[mbox, boxes] = point_sym_oobb(samples, points, coh, faces);
toc


[sbox, axes, bxid] = find_sym_box(boxes, thresh);

sym = [];
rot = [];


if(isempty(sbox))
    
    box = mbox;
    bx = process_bbox(box);
    
else
    
    box = reshape(sbox(1:end-3),3,5);
    bx = process_bbox(box);
    
    sym = axes;
    
    % check rotational sym
    
    rot = check_sym2(box, sym, mesh, samples, thresh);

    
end


bbox = struct('center', bx(:,1), 'origin', bx(:,2), 'axes', [box(:,2:4); 2*box(:,5)'], 'sym', axes, 'rot', rot);

bbox = calc_axis_dir(bbox);
%plot_box(bbox,1,1);



end



function [sbox, axes, bid] = find_sym_box(boxes, thresh)

%thresh = 1e-5; %1e-3;

sbox = [];
axes = [];
bid = -1;
%rots = [];

num = size(boxes,2);
min1 = 10;
min2 = 10;
min3 = 10;
min1_ind = -1;
min2_ind = -1;
min3_ind = -1;

for i=1:num
    ext1 = boxes(end-5,i);
    ext2 = boxes(end-4,i);
    ext3 = boxes(end-3,i);
    vol = ext1 * ext2 * ext3;
    score1 = boxes(end-2,i);
    score2 = boxes(end-1,i);
    score3 = boxes(end,i);
    
    if(vol < min1 && score1 < thresh)
        min1 = vol;
        min1_ind = [i 1];
    end
    
    if(vol < min1 && score2 < thresh)
        min1 = vol;
        min1_ind = [i 2];
    end
    
    if(vol < min1 && score3 < thresh)
        min1 = vol;
        min1_ind = [i 3];
    end
    
    if(vol < min2 && score1 < thresh && score2 < thresh)
        %min2 = (score1+score2)/2;
        min2 = vol;
        min2_ind = [i 1 2];
    end
    
    if(vol < min2 && score1 < thresh && score3 < thresh)
        %min2 = (score1+score3)/2;
        min2 = vol;
        min2_ind = [i 1 3];
    end
    
    if(vol < min2 && score2 < thresh && score3 < thresh)
        %min2 = (score2+score3)/2;
        min2 = vol;
        min2_ind = [i 2 3];
    end
    
    if(vol < min3 && score1 < thresh && score2 < thresh && score3 < thresh)
        %min3 = (score1+score2+score3)/3;
        min3 = vol;
        min3_ind = i;
    end
    
end

%return;

if(min3_ind ~= -1)
     sbox = boxes(:,min3_ind);
     axes = 1:3;
     bid = min3_ind;
%      rot1 = is_rot_sym(1, sbox, boxes, thresh);
%      rot2 = is_rot_sym(2, sbox, boxes, thresh);
%      rot3 = is_rot_sym(3, sbox, boxes, thresh);
%      
%      rots = [rot1, rot2, rot3];
%      nz = find(rots > 0);
%      rots = rots(nz);
     
     return;
end

if(min2_ind ~= -1)
     sbox = boxes(:,min2_ind(1));
     axes = min2_ind(2:3);
     bid = min2_ind(1);
%      rot_cand = setdiff(1:3, axes);
%      rot1 = is_rot_sym(rot_cand, sbox, boxes, thresh);
%      
%      rots = rot1;
%      nz = find(rots > 0);
%      rots = rots(nz);
     
     return;
end

if(min1_ind ~= -1)
     sbox = boxes(:,min1_ind(1));
     axes = min1_ind(2);
     bid = min1_ind(1);
     return;
end


end



function bx = process_bbox(box)

cen = box(:,1);
ext = box(:,5);
[sr_ext, sr_ind] = sort(ext, 'descend');
v1 = box(:,1+sr_ind(1));
v2 = box(:,1+sr_ind(2));
v3 = box(:,1+sr_ind(3));


v11 = v1*2*sr_ext(1);
v22 = v2*2*sr_ext(2);
v33 = v3*2*sr_ext(3);

%origin = cen - v1*ext(1) - v2*ext(2) - v3*ext(3);
origin = cen - v1*sr_ext(1) - v2*sr_ext(2) - v3*sr_ext(3);

bx = [cen origin v11 v22 v33];

end



