
function plot_box(box, show_axis, col)

    if ~exist('show_axis', 'var')
        show_axis = 0;
    end
    
    if ~exist('col', 'var')
        col = 0;
    end

    cols = {'r', 'g', 'b', 'y', 'm', 'c', 'k'};
    
    %origin = box.origin;
    center = box.center;
    exts = box.axes(end,:);
    hexts = 0.5*exts;
    axs = box.axes(1:3,:);
    origin = center - axs * hexts';
    
    v11 = box.axes(1:3, 1) * box.axes(4,1);
    v22 = box.axes(1:3, 2) * box.axes(4,2);
    v33 = box.axes(1:3, 3) * box.axes(4,3);
    
    p1 = origin;
    p2 = p1 + v11;
    p3 = p1 + v22;
    p4 = p1 + v33;
    p5 = p2 + v22;
    p6 = p2 + v33;
    p7 = p6 + v22;
    p8 = p7 - v11;
    
    hold on;
    
    plot3([p1(1); p2(1)], [p1(2); p2(2)], [p1(3); p2(3)]);
    plot3([p1(1); p3(1)], [p1(2); p3(2)], [p1(3); p3(3)]);
    plot3([p1(1); p4(1)], [p1(2); p4(2)], [p1(3); p4(3)]);

    plot3([p2(1); p5(1)], [p2(2); p5(2)], [p2(3); p5(3)]);
    plot3([p2(1); p6(1)], [p2(2); p6(2)], [p2(3); p6(3)]);

    plot3([p3(1); p5(1)], [p3(2); p5(2)], [p3(3); p5(3)]);
    plot3([p3(1); p8(1)], [p3(2); p8(2)], [p3(3); p8(3)]);

    plot3([p4(1); p6(1)], [p4(2); p6(2)], [p4(3); p6(3)]);
    plot3([p4(1); p8(1)], [p4(2); p8(2)], [p4(3); p8(3)]);

    plot3([p5(1); p7(1)], [p5(2); p7(2)], [p5(3); p7(3)]);

    plot3([p6(1); p7(1)], [p6(2); p7(2)], [p6(3); p7(3)]);

    plot3([p7(1); p8(1)], [p7(2); p8(2)], [p7(3); p8(3)]);

    if(col)
        f1 = [p1 p2 p5 p3];
        f2 = [p1 p2 p6 p4];
        f3 = [p1 p3 p8 p4];
        f4 = [p2 p5 p7 p6];
        f5 = [p3 p5 p7 p8];
        f6 = [p4 p6 p7 p8];

        if(isfield(box, 'label'))
            lbl = box.label;
        else
            lbl = 0;
        end
        
         if(isfield(box, 'part_id'))
            pid = box.part_id;
        else
            pid = 0;
        end
        
        %lbl = box.label;
        
        if(lbl == 0 || lbl > length(cols))
            c = rand([1,3]);
        else
            c = cols{lbl};
        end
        
        %alpha(0.5);
        h = fill3(f1(1,:), f1(2,:), f1(3,:), c);
        set(h, 'UserData', [pid lbl]);
        h = fill3(f2(1,:), f2(2,:), f2(3,:), c);
        set(h, 'UserData', [pid lbl]);
        h = fill3(f3(1,:), f3(2,:), f3(3,:), c);
        set(h, 'UserData', [pid lbl]);
        h = fill3(f4(1,:), f4(2,:), f4(3,:), c);
        set(h, 'UserData', [pid lbl]);
        h = fill3(f5(1,:), f5(2,:), f5(3,:), c);
        set(h, 'UserData', [pid lbl]);
        h = fill3(f6(1,:), f6(2,:), f6(3,:), c);
        set(h, 'UserData', [pid lbl]);
        
        dcm_obj = datacursormode(gcf);
        set(dcm_obj, 'UpdateFcn', @myupdatefcn);
    end
    
    if show_axis
        %center = box(:, 1);
        center = origin + 0.5*box.axes(1:3, 1)*box.axes(4,1) + 0.5*box.axes(1:3, 2)*box.axes(4,2) + 0.5*box.axes(1:3, 3)*box.axes(4,3);
        %axis1 = center + 0.5*box(:, 3);
        axis1 = center + box.axes(1:3, 1);
        axis2 = center + box.axes(1:3, 2);
        axis3 = center + box.axes(1:3, 3);

        plot3([center(1) axis1(1)], [center(2) axis1(2)], [center(3) axis1(3)], '-r');
        plot3([center(1) axis2(1)], [center(2) axis2(2)], [center(3) axis2(3)], '-g');
        plot3([center(1) axis3(1)], [center(2) axis3(2)], [center(3) axis3(3)], '-b');
    end
end


function output_txt = myupdatefcn(~, event_obj)
    pos = get(event_obj, 'Position');
    target = get(event_obj, 'Target');
    dt = get(target, 'UserData');
    %output_txt = {['X: ' num2str(pos(1))]; num2str(dt)};
    output_txt = {num2str(dt)};
end



