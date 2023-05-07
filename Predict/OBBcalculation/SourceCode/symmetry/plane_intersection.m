% Get the intersection point of three planes defined by three points and three normals
function point = plane_intersection(x1, x2, x3, n1, n2, n3)
% http://mathworld.wolfram.com/Plane-PlaneIntersection.html

    dv = det([n1 n2 n3]);
    if dv < eps
        point = [];
        return;
    end

    point = (dot(x1, n1)*cross(n2, n3) + ...
             dot(x2, n2)*cross(n3, n1) + ...
             dot(x3, n3)*cross(n1, n2)) / dv;
end
