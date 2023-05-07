%
% Transform cartersian coordinates into spherical coordinates without accouting for signed directions
%
% [theta, phi, r] = cart2sph(x, y, z)
%
% Input:
%   - x, y, z: cartersian coordinates
%
% Output:
%   - theta: polar angle (or inclination) in the range [0, pi/2]
%   - phi: azimuthal angle in the range [-pi, pi];
%   - r: radial distance
%
% The spherical coordinates are described in:
% http://en.wikipedia.org/wiki/Spherical_coordinates
% However, the coordinates are slightly changed to account for
% directions rather than signed directions (vectors). The parameter
% theta varies only between [0, pi/2] instead of [0, pi].
%
function [theta, phi, r] = cart2sph(x, y, z)

    % Get cartersian coordinates
    r = sqrt(x^2 + y^2 + z^2);
    theta = acos(z/r); 
    phi = atan2(y, x);

    % Ignore signed direction
    if theta > (pi/2)
        theta = acos(-z/r); 
        phi = atan2(-y, -x);

        if theta > (pi/2)
            disp('Error converting to spherical coordinates!');
        end
    end
end
