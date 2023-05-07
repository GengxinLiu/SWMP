%
% Exhaustively compute the symmetry score for a set of parameterized planes
%
% [score, pos_theta, pos_phi, samp] = sym_score_center(M, num_samples, num_bins)
%
% Input:
%   - M: triangle mesh
%   - num_samples: number of samples to be used in the computation of
%   the symmetry score
%   - num_bins: number of bins used for each dimension of the score
%   matrix (the plane parameterization)
%
% Output:
%   - score: score matrix of dimensions num_bins^2
%   - pos_theta, pos_phi: parameters of each cell in the matrix
%   - samp: samples used in the voting, of dimension <num_samples x 3>
%
% Parameterizes all possible planes in a 2D array and then computes the
% symmetry score for each plane. All the planes are assumed to cross the
% center of mass of the mesh
%
function [score, pos_theta, pos_phi, samp] = sym_score_center(M, num_samples, num_bins)
%
% The spherical coordinates used in the parameterization of normals are
% described in: http://en.wikipedia.org/wiki/Spherical_coordinates
% However, the coordinates are slightly changed to account for
% directions rather than signed directions (vectors). The parameter
% theta varies only between [0, pi/2] instead of [0, pi].

    % Compute triangle areas used for random sampling (to sample points from
    % the surface of the mesh)
    M = mesh_normalize(M);
    [N, A] = mesh_face_normals(M);

    % Compute samples for voting
    samp = mesh_sample(M, A, num_samples);

    % Range for each dimension in the accumulation matrix
    pos_theta = linspace(0, pi/2, num_bins); % Use only up to pi/2 to ignore signed directions
    pos_phi = linspace(-pi, pi, num_bins);

    % Score matrix: (theta, phi). theta and phi are the spherical
    % description of the planes' normal (with radius 1, since the
    % normals are normalized). All the planes are assumed to cross the
    % center of mass of the mesh. So, the distance of the plane to the
    % origin is derived from that.
    score = zeros(num_bins, num_bins);

    % Get center of gravity of the shape and use it as the point where
    % the planes pass through
    center = mean(samp);

    % Compute the symmetry score for each plane
    n = [0 0 0];
    r = 1; % Vectors are always normalized
    for i = 1:num_bins
        theta = pos_theta(i);
        for j = 1:num_bins
            phi = pos_phi(j);
            % Map spherical coordinates to cartesian
            [n(1), n(2), n(3)] = sph2cart(theta, phi, r);
            % Get distance of the plane to the origin
            d = -dot(n, center);
            % Compute sum of distances of reflected points to the mesh
            dist = sym_plane(samp, [n(1) n(2) n(3) d], M.vertices, M.faces);
            score(i, j) = norm(dist(:, 1));
        end
    end

    % Transform sum of distance into a score where larger numbers
    % indicate more symmetric planes
    score = max(score(:)) - score;
end
