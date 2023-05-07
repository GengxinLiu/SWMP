%
% Perform voting to find the dominant reflection planes in a mesh
%
% [vote, pos_theta, pos_phi, pos_d, samp] = sym_voting(M, num_samples, num_bins)
%
% Input:
%   - M: triangle mesh
%   - num_samples: number of samples to be used in the voting. The
%   voting will consider all the num_samples^2 pairs
%   - num_bins: number of bins used for each dimension of the
%   accumulation matrix. Should not be more than the number of samples
%
% Output:
%   - vote: accumulation matrix of dimensions num_bins^3
%   - pos_theta, pos_phi, pos_d: parameters of each cell in vote
%   - samp: samples used in the voting, of dimension <num_samples x 3>
%
function [vote, pos_theta, pos_phi, pos_d, samp] = sym_voting(M, num_samples, num_bins)
%
% This is a simplified version of the algorithm in Section 4.3 of:
% J. Podolak, P.  Shilane, A. Golovinskiy, S. Rusinkiewicz, T.
% Funkhouser, "A planar-reflective symmetry transform for 3D shapes",
% SIGGRAPH 2006. We only use points on the surface of the mesh to vote
% for reflections, rather than points on the full volume. 
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
    pos_d = linspace(-sqrt(12), sqrt(12), num_bins);
    % sqrt(12) is the diagonal in a normalized box where each axis
    % varies between [-1, 1]

    % Accumulation matrix: (theta, phi, r). theta and phi are the spherical
    % description of the planes' normals (with radius 1, since the normals
    % are normalized), and d is the negative distance of the origin to
    % the plane using the common equation a*x + b*y + c*z + d = 0
    vote = zeros(num_bins, num_bins, num_bins);

    % Set sampling parameter sigma (only needed if sampling point on the
    % whole volume)
    sigma = 0.01;

    % Collect 'num_samples' pairs of points on the surface of the mesh
    % and vote for the plane between each pair
    for i = 1:num_samples
        for j = 1:num_samples
            % Get pair of samples on the mesh
            P1 = samp(i, :); 
            P2 = samp(j, :);

            % Compute line between two points, which will be the plane's normal
            n = P2 - P1;

            % Compute line's midpoint, a point that lies on the plane
            midpoint = (P1 + P2) / 2;

            % Normalize normal, but save its norm to use it as a weight in the
            % voting
            dist = norm(n, 2);
            n = n ./ dist;

            % Convert normal to spherical system
            [theta, phi, r] = cart2sph(n(1), n(2), n(3));

            % Transform back, to get a direction (instead of a signed
            % direction). Without this step, we may mess up d
            [n(1), n(2), n(3)] = sph2cart(theta, phi, r);

            % Compute d
            d = -n(1)*midpoint(1) -n(2)*midpoint(2) -n(3)*midpoint(3);
            % Check if value falls outside our voting space
            if d > sqrt(12)
                disp('Error: d > sqrt(12)');
            end

            % Add plane to the accumulation matrix
            % Find cell indices
            p1 = find(pos_theta > theta);
            if isempty(p1), p1 = 1; end

            p2 = find(pos_phi > phi);
            if isempty(p2), p2 = 1; end

            p3 = find(pos_d > d);
            if isempty(p3), p3 = 1; end

            % Add vote to the matrix
            vote(p1(1), p2(1), p3(1)) = vote(p1(1), p2(1), p3(1)) + vote_value(M, P1, P2, dist, theta, sigma);
        end
    end

end


% Compute sampling vote value according to mesh function and weight
function v = vote_value(M, P1, P2, dist, theta, sigma)

    %d1 = distfunc(M, P1);
    %d2 = distfunc(M, P2);
    %% v = sampling_weight * f(P1) * f(P2)
    %v = (1/(meshfunc(M, P1, sigma, d1)*meshfunc(M, P2, sigma, d2)*2*dist^2*sin(theta))) * d1 * d2;

    % Use a simplified version of the weight in the paper cited above
    % Instead of considering the volume, we only consider points on the
    % mesh, which makes f(x) = 1 in the formula above
    v = 1/(2*dist^2*sin(theta));
end


% Weight mesh distance with a Gaussian
function d = meshfunc(M, P, sigma, dist)

    d = exp(-dist/sigma^2);
end


% Compute the minimum distance from vP to all the vertices of M
function d = distfunc(M, P)

    diff = (repmat(P, size(M.vertices, 1), 1) - M.vertices).^2;
    diff = sum(diff')';
    diff = sqrt(diff);
    d = min(diff);
end
