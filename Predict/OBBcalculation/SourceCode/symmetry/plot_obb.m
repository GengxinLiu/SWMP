% Plot an obb and its axes
function plot_obb(mesh, obb);

    mesh_show_color(mesh);
    hold on;

    origin = obb(2, :);
    %axis1 = obb(2, :)*obb(5, 1);
    %axis2 = obb(3, :)*obb(5, 2);
    %axis3 = obb(4, :)*obb(5, 3);
    axis1 = obb(3, :);
    axis2 = obb(4, :);
    axis3 = obb(5, :);

    plot_line(origin, origin+axis1);
    plot_line(origin, origin+axis2);
    plot_line(origin+axis1, origin+axis1+axis2);
    plot_line(origin+axis2, origin+axis1+axis2);

    plot_line(origin+axis3, origin+axis3+axis1);
    plot_line(origin+axis3, origin+axis3+axis2);
    plot_line(origin+axis3+axis1, origin+axis3+axis1+axis2);
    plot_line(origin+axis3+axis2, origin+axis3+axis1+axis2);

    plot_line(origin, origin+axis3);
    plot_line(origin+axis1, origin+axis1+axis3);
    plot_line(origin+axis2, origin+axis2+axis3);
    plot_line(origin+axis1+axis2, origin+axis1+axis2+axis3);

    plot3([0 axis1(1)], [0 axis1(2)], [0 axis1(3)], '-r');
    plot3([0 axis2(1)], [0 axis2(2)], [0 axis2(3)], '-g');
    plot3([0 axis3(1)], [0 axis3(2)], [0 axis3(3)], '-b');
end


function plot_line(a, b)

    plot3([a(1) b(1)], [a(2) b(2)], [a(3) b(3)], '-b');
end
