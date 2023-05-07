
function is_sym = is_sym_plane(dist, thresh)


    dist = dist(:,1);
    sorted_dist = sort(dist);
    len = length(sorted_dist);
    if(sorted_dist(round(0.9*len)) < thresh)
        is_sym = 1;
    else
        is_sym = 0;
    end

end