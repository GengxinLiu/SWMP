%
% Sample points uniformly accross a mesh
%
% [sample [, normal]] = mesh_sample(M, A, num_samples [, N])
%
% Input:
%   - M: input mesh, as returned by msh_read_smf()
%   - A: face areas, as returned by mesh_face_normals()
%   - num_samples: number of samples to collect
%   - N (optional): face normals, as returned by mesh_face_normals().
%   This parameter has to be supplied if the output argument 'normal' is
%   requested.
%
% Output:
%   - sample: set of samples, where sample(i, :) are the 3D coordinates
%   of sample point 'i'
%   - normal (optional): normal vectors, where normal(i, :) is the 3D
%   normal vector at sample point 'i'. These normals are directly
%   derived from the face normals.
%
% See also mesh_face_normals
%
function [sample, normal] = mesh_sample(M, A, num_samples, N)
%
% Copyright (c) 2008 Oliver van Kaick <ovankaic@cs.sfu.ca>
%

    % Initialize sample
    sample = zeros(num_samples, 3);

    % Check if normals were supplied
    if (nargout > 1) && (nargin < 4)
        disp('Error: sample normals were requested, but no face normals are provided!');
        return;
    end

    % Initialize normals, if requested
    copy_normals = 0;
    if nargout > 1
        normal = zeros(num_samples, 3);
        copy_normals = 1;
    end

    % Create cumulative sum based on triangle areas
    % Transform areas into a probability distribution
    A = A ./ sum(A);
    % Compute cumulative sum of probability distribution
    S = cumsum(A);

    % Sample random points
    for i = 1:num_samples
        % First, randomly sample a triangle, with a probability given by its
        % areas
        % Get a random number
        r = rand();
        % Find location in the cumulative sum where entry is larger than the
        % random number
        tri = find(S > r, 1);
        % Now, sample a random point in this triangle
        sample(i, :) = random_triangle_sample(M, tri);
        % Copy normal vector, if requested
        if copy_normals
            normal(i, :) = N(tri, :);
        end
    end
end


% Randomly sample a point inside of a triangle
function point = random_triangle_sample(M, indx)

%    s = rand();
%    t = rand();
%
%    if (s + t) > 1
%        s = 1 - s;
%        t = 1 - t;
%    end
%
%    v0 = M.vertices(M.faces(indx, 1), :);
%    v1 = M.vertices(M.faces(indx, 2), :);
%    v2 = M.vertices(M.faces(indx, 3), :);
%
%    point = s*v0 + t*v1 + (1-s-t)*v2;

    % Unbiased version
    % See http://stackoverflow.com/questions/4778147/sample-random-point-in-triangle
    r1 = rand();
    r2 = rand();

    v0 = M.vertices(M.faces(indx, 1), :);
    v1 = M.vertices(M.faces(indx, 2), :);
    v2 = M.vertices(M.faces(indx, 3), :);

    point = (1 - sqrt(r1))*v0 + sqrt(r1)*(1 - r2)*v1 + sqrt(r1)*r2*v2;
end
