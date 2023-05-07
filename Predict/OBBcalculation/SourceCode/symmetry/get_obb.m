%
% Get oriented bounding box from a set of three orthogonal axes
%
% box = get_obb(n1, n2, n3, mesh)
%
% Input:
%   - n1, n2, n3: three orthogonal axes (the vectors should be
%   normalized and of dimensions <1 x 3>)
%   - mesh: mesh for which the bounding box should be computed
%
% Output:
%   - box: a <6 x 3> matrix with the following 1x3 rows:
%          [<center>; <origin>; <axis 1>; <axis 2>; <axis 3>; <extent along each axis>]
%     Note that the axes are already scaled by the extent values.
%
function box = get_obb(n1, n2, n3, mesh)

    % Check size
    if any(size(n1) ~= [1 3])
        disp('The axes need to be <1 x 3> vectors');
    end
    if any(size(n2) ~= [1 3])
        disp('The axes need to be <1 x 3> vectors');
    end
    if any(size(n3) ~= [1 3])
        disp('The axes need to be <1 x 3> vectors');
    end

    % Negate the axes if the contrary direction has less negative signs
    if 1
    if length(find(-n1 > 0)) > length(find(n1 > 0))
        n1 = -n1;
    end
    if length(find(-n2 > 0)) > length(find(n2 > 0))
        n2 = -n2;
    end
    if length(find(-n3 > 0)) > length(find(n3 > 0))
        n3 = -n3;
    end
    end

    % Get projection of the mesh unto each axis
    [mn1, mx1] = axis_dim(n1, mesh);
    [mn2, mx2] = axis_dim(n2, mesh);
    [mn3, mx3] = axis_dim(n3, mesh);

    % Get vectors pointing to the origin
    vec1 = mn1*n1;
    vec2 = mn2*n2;
    vec3 = mn3*n3;

    % Get origin
    origin = vec1 + vec2 + vec3;

    % Get center
    %vec1 = ((mx1 + mn1)/2)*n1;
    %vec2 = ((mx2 + mn2)/2)*n2;
    %vec3 = ((mx3 + mn3)/2)*n3;
    %center = vec1 + vec2 + vec3;

    % Get extent of each axis
    extent = [mx1 - mn1, mx2 - mn2, mx3 - mn3];

    % Scale axes by extent
    n1 = extent(1)*n1;
    n2 = extent(2)*n2;
    n3 = extent(3)*n3;

    % Get center
    center = origin + 0.5*n1 + 0.5*n2 + 0.5*n3;

    % Sort axes by their norm
    [val, ps] = sort([norm(n1, 2) norm(n2, 2) norm(n3, 2)], 'descend');
    n = [n1; n2; n3];
    n = n(ps, :);

    % Set output
    box = [center; origin; n(1, :); n(2, :); n(3, :); extent]; 
end


% Project mesh unto n and find minimum and maximum projection values
function [mn, mx] = axis_dim(n, mesh)

    dp = repmat(n, size(mesh.vertices, 1), 1) .* mesh.vertices;
    dp = sum(dp')';
    mn = min(dp);
    mx = max(dp);
end
