%function [g, f] = check_sym2(obb, sym, mesh, samp, thresh)
function g = check_sym2(obb, sym, mesh, samp, thresh)

    g = [];
    
    if(length(sym) < 2)
        return;
    end

    for i=1:length(sym)
        for j=i+1:length(sym)
            n1 = sym(i);
            n2 = sym(j);
            is_rot = check_planes(obb(:,n1+1), obb(:,n2+1), obb(:,1), mesh, samp, thresh);
            if(is_rot)
                g = [g; setdiff(1:3, [n1 n2])];
            end
        end
    end
    
%     for i = 2:4
%         if(isempty(find(sym == i-1)))
%             continue;
%         end
%         g(i-1) = check_plane(obb(:, i), obb(:, setdiff(2:4, i)), mesh, samp, thresh);
%     end

    %f(1) = (g(2) > (10*g(1))) || (g(3) > (10*g(1)));
    %f(2) = (g(1) > (10*g(2))) || (g(3) > (10*g(2)));
    %f(3) = (g(1) > (10*g(3))) || (g(2) > (10*g(3)));
end


function g = check_planes(n1, n2, cen, mesh, samp, thresh)

g = 0;

tol = 0.01;
v = zeros(3, 6);
v(:, 1) = n1 + n2;
v(:, 2) = n1 + 0.5*n2;
v(:, 3) = 0.5*n1 + n2;
v(:, 4) = n1 - n2;
v(:, 5) = n1 - 0.5*n2;
v(:, 6) = 0.5*n1 - n2;

for i=1:size(v,2)
    v(:, i) = v(:, i) / norm(v(:, i));
end


d = zeros(1, size(v,2));

for i=1:size(v,2)
    di = sum(v(:,i) .* cen);
    d(i) = -di;
end

% Check symmetry score for four variations

is_sym = zeros(size(v,2), 1);
for i = 1:size(v,2)
    dist = sym_plane(samp, [v(:, i)' d(i)], mesh.vertices, mesh.faces);
    
    is_sym(i) = is_sym_plane(dist, thresh);
    
end

zers = find(is_sym == 0);
if(isempty(zers))
    g = 1;
end

end


function g = check_planes_old(n1, n2, cen, mesh, samp, thresh)

g = 0;

tol = 0.01;
v = zeros(3, 4);
v(:, 1) = n1 + tol*n2;
v(:, 2) = n1 - tol*n2;
v(:, 3) = n1 + 2*tol*n2;
v(:, 4) = n1 - 2*tol*n2;

d = zeros(1, 4);

for i=1:4
    di = sum(v(:,i) .* cen);
    d(i) = -di;
end

% Check symmetry score for four variations
%score = zeros(4, 1);
is_sym = zeros(4, 1);
for i = 1:4
    dist = sym_plane(samp, [v(:, i)' d(i)], mesh.vertices, mesh.faces);
    %score(i) = norm(dist(:, 1));
    is_sym(i) = is_sym_plane(dist, thresh);
%     dist = dist(:,1);
%     sorted_dist = sort(dist);
%     len = length(sorted_dist);
%     if(sorted_dist(round(0.9*len)) < thresh)
%         is_sym(i) = 1;
%     end
    
end

zers = find(is_sym(1:4) == 0);
if(isempty(zers))
    g = 1;
end

end


% Perturb the normal along each axis and see if the symmetry is still
% maintained
function g = check_plane(n, ax, mesh, samp, thresh)

    g = 0;
    
    % Normalize axis
    n = n ./ norm(n, 2);
    ax(:, 1) = ax(:, 1) / norm(ax(:, 1), 2);
    ax(:, 2) = ax(:, 2) / norm(ax(:, 2), 2);

    % Get two linearly independent vectors that span the plane
    %T = eye(3) - n*n';
    %[v, e] = eig(T);

    % Get four variations of the normal
%    tol = 0.01;
%    v = zeros(3, 4);
%    v(:, 1) = n + tol*v(:, 2);
%    v(:, 2) = n - tol*v(:, 2);
%    v(:, 3) = n + tol*v(:, 3);
%    v(:, 4) = n - tol*v(:, 3);

    %(180*acos(angle))/pi
    % Threshold 0.01 is for 10 degrees variation
    tol = 0.01;
    v = zeros(3, 4);
    v(:, 1) = n + tol*ax(:, 1);
    v(:, 2) = n - tol*ax(:, 1);
    v(:, 3) = n + tol*ax(:, 2);
    v(:, 4) = n - tol*ax(:, 2);

    % Check symmetry score for four variations
    score = zeros(5, 1);
    is_sym = zeros(5, 1);
    for i = 1:4
        dist = sym_plane(samp, [v(1:3, i)' 0], mesh.vertices, mesh.faces);
        score(i) = norm(dist(:, 1));
        
        sorted_dist = sort(dist);
        len = length(sorted_dist);
        if(sorted_dist(round(0.9*len)) < thresh)
            is_sym(i) = 1;
        end
    end
    dist = sym_plane(samp, [n' 0], mesh.vertices, mesh.faces);
    score(5) = norm(dist(:, 1));
    sorted_dist = sort(dist);
    len = length(sorted_dist);
    if(sorted_dist(round(0.9*len)) < thresh)
        is_sym(5) = 1;
    end
    
    %g1 = abs(score(1) - score(5)) + abs(score(2) - score(5));
    %g2 = abs(score(3) - score(5)) + abs(score(4) - score(5));
    %g = min([g1 g2]);
    
    zers = find(is_sym(1:4) == 0);
    if(isempty(zers))
        g = 1;
    end
end
