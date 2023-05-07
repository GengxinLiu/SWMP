%
% Compute an OOBB of a set of 3D points
%
% Input:
%   - points: a <n x 3> matrix containing the 3D input points
%
% Output:
%   - box: a <3 x 5> matrix with the following 3x1 columns:
%          [<center> <axis 1> <axis 2> <axis 3> <extent along each axis>]
%
function box = point_oobb(points)

    if size(points, 2) ~= 3
        error('Invalid dimension for point set. The input should be a <n x 3> matrix');
    end

    if size(points, 1) <= 0
        error('Point set should have at least one point');
    end

    box = oobb(points);
end
