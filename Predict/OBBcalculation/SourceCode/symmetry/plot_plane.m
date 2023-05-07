%
% Plot a plane given by its normal and a point lying on the plane
% The normal and point have to be 3x1 vectors!
%
% xLim and yLim are the extent of the plane, each given as a pair [min
% max], which are factors that multiply two vectors with norm 1
%
% Example: plot_plane([1 0 0]', [0 0 0]', [-1 1], [-1 1], 'b');
%
function plot_plane(normal, point, xLim, yLim, color)

    % Get two linearly independent vectors that span the plane
    T = eye(3) - normal*normal';
    [v, e] = eig(T);

    % Get four vertices of the plane given by a combination of the two
    % vectors that span the plane
    v1 = xLim(1)*v(:, 2) + yLim(1)*v(:, 3) + point;
    v2 = xLim(1)*v(:, 2) + yLim(2)*v(:, 3) + point;
    v3 = xLim(2)*v(:, 2) + yLim(2)*v(:, 3) + point;
    v4 = xLim(2)*v(:, 2) + yLim(1)*v(:, 3) + point;

    % Plot two triangles
    %fill3([v1(1) v2(1) v4(1)], [v1(2) v2(2) v4(2)], [v1(3) v2(3) v4(3)], color);
    %fill3([v1(1) v4(1) v3(1)], [v1(2) v4(2) v3(2)], [v1(3) v4(3) v3(3)], color);
    fill3([v1(1) v2(1) v3(1) v4(1)], [v1(2) v2(2) v3(2) v4(2)], [v1(3) v2(3) v3(3) v4(3)], color);
end
