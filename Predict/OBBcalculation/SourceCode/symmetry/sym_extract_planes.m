%
% Extract three orthogonal planes from a matrix of symmetry scores
%
% [C, val, samp] = sym_extract_planes(M, score, pos_theta, pos_phi, num_samples)
%
% Input:
%   - score, pos_theta, pos_phi: result as returned by sym_score_center
%   - num_samples: number of samples to use in the computation of the
%   symmetry score. If a matrix is given instead of a scalar, it is
%   taken as the matrix of samples
%
% Output:
%   - C(axis, :): three dominant symmetry axes, each a vector of
%   dimensions <1 x 3>
%   - val: score value for each axis
%   - samp: samples used for the computation of the symmetry score
%
function [C, val, samp] = sym_extract_planes2(M, score, pos_theta, pos_phi, num_samples)

    % Init output
    C = zeros(3, 4);
    val = zeros(3, 1);

    % Get samples to evaluate symmetry score
    if (size(num_samples, 1) > 1) && (size(num_samples, 2) == 3)
        samp = num_samples;
        num_samples = size(samp, 1);
    else
        [N, A] = mesh_face_normals(M);
        samp = mesh_sample(M, A, num_samples);
    end

    % Get center of gravity of the shape
    center = mean(samp);

    % Find first plane
    flag = ones(prod(size(score)), 1);
    indx = select_plane(score, flag);
    % Retrieve selected plane
    [x, y] = ind2sub(size(score), indx(1));
    temp = [pos_theta(x) pos_phi(y) 1];
    [C(1, 1), C(1, 2), C(1, 3)] = sph2cart(temp(1), temp(2), 1);
    % Place the plane at the center of gravity of the shape
    C(1, 4) = -dot(C(1, 1:3), center);
    % Keep score
    val(1) = score(x, y);
    % Never consider this plane again
    score(x, y) = 0;

    % Find second plane
    % Consider only planes that are orthogonal to the first plane
    % already found

    % Mark planes that are orthogonal to the first plane with a flag
    flag = zeros(prod(size(score)), 1);
    tol = 0.1; % Tolerance in the angle for orthogonality 
    [v, indx] = sort(score(:), 'descend');
    for i = 1:length(flag)
        % Get plane
        [x, y] = ind2sub(size(score), indx(i));
        temp = [pos_theta(x) pos_phi(y) 1];
        [temp_c(1), temp_c(2), temp_c(3)] = sph2cart(temp(1), temp(2), 1);
        % Check plane's angle with the first plane
        if (vec_angle(temp_c, C(1, 1:3)) > ((pi/2)*(1-tol))) && ...
           (vec_angle(temp_c, C(1, 1:3)) < ((pi/2)*(1-tol) + (pi/2)*(2*tol)))
            flag(indx(i)) = 1;
            % We only need one plane
            break;
        end
    end

    % Retrieve selected plane
    indx = select_plane(score, flag);
    [x, y] = ind2sub(size(score), indx(1));
    temp = [pos_theta(x) pos_phi(y) 1];
    [C(2, 1), C(2, 2), C(2, 3)] = sph2cart(temp(1), temp(2), 1);
    %C(2, 4) = temp(3);
    C(2, 4) = -dot(C(2, 1:3), center);
    val(2) = score(x, y);
    score(x, y) = 0;

    % Find third plane
    % Consider only planes that are orthogonal to the first and second
    % planes
    flag = zeros(prod(size(score)), 1);
    [v, indx] = sort(score(:), 'descend');
    for i = 1:length(flag)
        [x, y] = ind2sub(size(score), indx(i));
        temp = [pos_theta(x) pos_phi(y) 1];
        [temp_c(1), temp_c(2), temp_c(3)] = sph2cart(temp(1), temp(2), 1);
        if (vec_angle(temp_c, C(1, 1:3)) > ((pi/2)*(1-tol))) && ...
           (vec_angle(temp_c, C(1, 1:3)) < ((pi/2)*(1-tol) + (pi/2)*(2*tol))) && ...
           (vec_angle(temp_c, C(2, 1:3)) > ((pi/2)*(1-tol))) && ...
           (vec_angle(temp_c, C(2, 1:3)) < ((pi/2)*(1-tol) + (pi/2)*(2*tol)))
            flag(indx(i)) = 1;
            break;
        end
    end

    % Retrieve selected plane
    indx = select_plane(score, flag);
    [x, y] = ind2sub(size(score), indx(1));
    temp = [pos_theta(x) pos_phi(y) 1];
    [C(3, 1), C(3, 2), C(3, 3)] = sph2cart(temp(1), temp(2), 1);
    %C(3, 4) = temp(3);
    C(3, 4) = -dot(C(3, 1:3), center);
    val(3) = score(x, y);
    score(x, y) = 0;

    %vec_angle(C(1, 1:3), C(2, 1:3)) 
    %vec_angle(C(1, 1:3), C(3, 1:3)) 
    %vec_angle(C(2, 1:3), C(3, 1:3)) 
    %C

    % Make the axes perfectly orthogonal and normalized
    C(3, 1:3) = cross(C(1, 1:3), C(2, 1:3));
    C(2, 1:3) = cross(C(1, 1:3), C(3, 1:3));
    C(3, 1:3) = C(3, 1:3) ./ norm(C(3, 1:3), 2);
    C(2, 1:3) = C(2, 1:3) ./ norm(C(2, 1:3), 2);

    % Get point at which all the planes intersect as the point common to
    % all the planes
    p1 = plane_point(C(1, 1:3)', C(1, 4));
    p2 = plane_point(C(2, 1:3)', C(2, 4));
    p3 = plane_point(C(3, 1:3)', C(3, 4));
    point = plane_intersection(p1, p2, p3, C(1, 1:3)', C(2, 1:3)', C(3, 1:3)');
    if ~isempty(point)
        %disp('Found intersection');
        %C(1, 1:3) = cross(C(2, 1:3), C(3, 1:3));
        %C(2, 1:3) = cross(C(1, 1:3), C(3, 1:3));
        C(1, 4) = -C(1, 1)*point(1) - C(1, 2)*point(2) - C(1, 3)*point(3);
        C(2, 4) = -C(2, 1)*point(1) - C(2, 2)*point(2) - C(2, 3)*point(3);
        C(3, 4) = -C(3, 1)*point(1) - C(3, 2)*point(2) - C(3, 3)*point(3);
    end

    % Debug
    if 0
        figure;
        mesh_show_color(M);
        hold on;
        %[xx,yy,zz] = meshgrid(-5:0.1:5, -5:0.1:5, -5:0.1:5);
        %clr = {'red'; 'green'; 'blue'};
        clr = 'rgb';
        for pix = 1:3
            %theta = C(pix, 1);
            %phi = C(pix, 2);
            %d = C(pix, 3);
            %[a, b, c] = sph2cart(theta, phi, 1);
            a = C(pix, 1);
            b = C(pix, 2);
            c = C(pix, 3);
            d = C(pix, 4);
            plot_plane([a b c]', [a*-d b*-d c*-d]', [-1 1], [-1 1], clr(pix));
            %fv = isosurface(xx, yy, zz, a*xx+b*yy+c*zz+d, 0);
            %p = patch(fv);
            %set(p, 'FaceColor', clr{pix}, 'EdgeColor', 'none');
        end
        %plot3(smpl(:, 1), smpl(:, 2), smpl(:, 3), '+b');
    end
end


% Get the plane with the largest symmetry score that also has flag == 1
function indx = select_plane(score, flag)

    sel = find(flag == 1);
    [mn, ps] = max(score(sel));
    indx = sel(ps(1));
end
