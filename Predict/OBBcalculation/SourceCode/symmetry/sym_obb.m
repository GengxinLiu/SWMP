% 
% Compute OBB for a mesh based on symmetry axes
%
% obb = sym_obb(mesh [, debug])
%
% Input:
%   - mesh: mesh for which the OBB should be computed
%   - debug (optional): if 1, print timing, if 2, plot obb
%
% Output:
%   - obb: OBB for the mesh, as defined in get_obb.m
%
% See also get_obb
%
function obb = sym_obb(mesh, debug)

    if ~exist('debug', 'var')
        debug = 0;
    end

    % Perform voting of symmetry planes
    num_samples = 50;
    num_bins = 50;
    if debug > 0
        tic
    end
    [score, pos_theta, pos_phi, samp] = sym_score_center(mesh, num_samples, num_bins);
    if debug > 0
        toc
    end

    % Collect top symmetries
    if debug > 0
        tic
    end
    [C, val] = sym_extract_planes(mesh, score, pos_theta, pos_phi, samp);
    if debug > 0
        toc
    end

    % Get OBB from symmetry planes
    obb = get_obb(C(1,1:3), C(2,1:3), C(3,1:3), mesh);

    g = check_sym2(obb, mesh, samp);
    g

    % Debug
    if debug > 1
        figure; plot_obb(mesh, obb);
    end
end
