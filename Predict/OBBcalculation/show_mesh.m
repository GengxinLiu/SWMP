%
% This function displays a triangle mesh
%
% function ph = show_mesh(M [, show_edges [, show_color]])
%
% Input -
%   - M: triangle mesh: M.vertices(i, :) represents the 3D coordinates
%   of vertex 'i', while M.faces(i, :) contains the indices of the three
%   vertices that compose face 'i'. If 'FaceVertexCData' is defined, it
%   is used as the color attributes of the mesh
%   - show_edges (optional): if set to 1, triangle edges are shown in
%   black (default value is 0)
%   - show_color (optional): if set to 0, FaceVertexCData is ignored
%   (default value is 1)
%
% Output -
%   - ph: handle for patch object
%   - lh: handle for camera light object
%
function ph = show_mesh(M, show_edges, show_color)
%
% Copyright (c) 2008, 2009, 2010 Oliver van Kaick <ovankaic@cs.sfu.ca>
%
% Modified by Ruizhen Hu 
%

% Check input arguments
if nargin <= 1
    show_edges = 0;
end

if nargin <= 2
    show_color = 1;
end
 
% Plot mesh
ph = trimesh(M.faces, M.vertices(:,1), M.vertices(:,2), M.vertices(:,3));

% Set colors
if (isfield(M, 'FaceVertexCData')) && show_color
    if size(M.FaceVertexCData, 1) == size(M.faces, 1)
        % Face interpolation can only be flat
        set(ph, 'FaceColor', 'flat');
    else
        % Vertex colors should be interpolated across the faces
        set(ph, 'FaceColor',' interp');
    end
else
	% Transparent faces 
    set(ph, 'FaceAlpha', 0);
    set(ph, 'EdgeAlpha', 0.8);
    set(ph, 'EdgeColor', [0.85 0.85 0.85]);
    set(ph, 'LineWidth', 0.01);
end

% Set edge colors
if ~show_edges
    set(ph, 'EdgeColor', 'none');
end

% Set aspect ratio
daspect([1 1 1]);

% Set axis
axis tight;
axis off;

% Set lightning
% lh = camlight;
% lighting gouraud;
% set(ph, 'AmbientStrength', 0.6, 'SpecularStrength', 1);
% lighting phong;
% light('Position',[1 0 0],'Style','infinite');
% light('Position',[-1 0 0],'Style','infinite');
% light('Position',[0 1 0],'Style','infinite');
% light('Position',[0 0 -1],'Style','infinite');

% Set better rotation mode
%h = cameratoolbar('setmode', 'orbit');

% Set view
%view(2); 

% Update lighting
% h = rotate3d;
%set(h, 'ActionPostCallback',{@movelight, lh});
%set(h,'Enable','on');
%set(h,'UserData',lh);

%function movelight(src, eventdata, lh)
%camlight(lh, 'headlight');
%drawnow;
