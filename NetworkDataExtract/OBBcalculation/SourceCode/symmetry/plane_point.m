% Get point closest to the plane defined by n(1)*x + n(2)*y + n(3)*z + d = 0
function point = plane_point(n, d);

    point = n*-d;
end
