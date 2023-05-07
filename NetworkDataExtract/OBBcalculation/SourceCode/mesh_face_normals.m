%
% Compute the normals at the mesh faces and the area of the faces
%
% [N, A] = mesh_face_normals(M)
%
% Input -
%   - M: input mesh (as given by msh_read_smf())
%
% Output -
%   - N: normals: N(i, :) is the 3D normal at face 'i' (the normals are
%   all unit-normalized)
%   - A: areas: A(i) is the area of face 'i'
%
% See also mesh_read_smf, mesh_vertex_normals, mesh_show_face_normals
%
function [N, A] = mesh_face_normals(M)
%
% Copyright (c) 2008 Oliver van Kaick <ovankaic@cs.sfu.ca>
%

% Allocate matrix to store normals
N = zeros(size(M.faces, 1), 3);
A = zeros(size(M.faces, 1), 1);

% Compute normals
for i = 1:size(M.faces, 1)
    % Get face vertices
    v(1, :) = M.vertices(M.faces(i, 1), :);
    v(2, :) = M.vertices(M.faces(i, 2), :);
    v(3, :) = M.vertices(M.faces(i, 3), :);

    % Compute face normal
    v0 = v(2, :) - v(1, :);
    v1 = v(3, :) - v(1, :);
    n = cross(v0, v1);
    area = norm(n, 2)/2.0;
    n = n/norm(n, 2);

    % Assign face normal
    N(i, :) = n;

    % Assign area
    A(i) = area;
end
