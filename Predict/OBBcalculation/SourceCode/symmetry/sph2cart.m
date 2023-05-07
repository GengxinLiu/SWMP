%
% Transform spherical coordinates into cartersian coordinates without accouting for signed directions
%
% [x, y, z] = sph2cart(theta, phi, r)
%
% Input:
%   - theta: polar angle (or inclination) in the range [0, pi/2]
%   - phi: azimuthal angle in the range [-pi, pi];
%   - r: radial distance
%
% Output:
%   - x, y, z: cartersian coordinates
%
% The spherical coordinates are described in:
% http://en.wikipedia.org/wiki/Spherical_coordinates
% However, the coordinates are slightly changed to account for
% directions rather than signed directions (vectors). The parameter
% theta varies only between [0, pi/2] instead of [0, pi].
%
function [x, y, z] = sph2cart(theta, phi, r)
        
    x = r*sin(theta)*cos(phi);
    y = r*sin(theta)*sin(phi);
    z = r*cos(theta);
end
