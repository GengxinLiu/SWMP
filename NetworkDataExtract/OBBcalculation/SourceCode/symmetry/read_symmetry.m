%
% Read a symmetry file generated by the symmetry analysis software
%
% rel = read_symmetry(filename)
%
% Input:
%   - filename: name of the symmetry file
%
% Output:
%   - rel: symmetry relations. A structure with three fields: 
%       - seg: the segments involved in the symmetry, starting at index 1
%       - name: name of the symmetry type (a string). One of 'trans' or 'ref'
%       - v: vector of symmetry parameters (real values). The number of
%       parameters depends on the symmetry type. Symmetry types:
%           - trans: translation. Given by a vector (x, y, z)
%           quantifying the amount of translation along each axis
%           - ref: reflection. Given by a plane normal (v1, v2, v3) and
%           a plane point (x, y, z) describing the plane of reflection
%           - rot: rotation. Given by an axis direction (v1, v2, v3), an
%           axis point (x, y, z), and an angle (alpha)
%
function rel = read_symmetry(filename)

    fid = fopen(filename, 'r');

    rel = struct([]);

    while ~feof(fid)
        c = fscanf(fid, '%c', 1);
        while isspace(c)
            c = fscanf(fid, '%c', 1);
        end
        if feof(fid)
            return;
        end
        if c == '<'
            p = fscanf(fid, '(%d, %d), ');
            rel(end+1).seg = p' + 1;
            name = fscanf(fid, '%s, ');
            c = fscanf(fid, '%c', 1);
            while isspace(c)
                c = fscanf(fid, '%c', 1);
            end
            if strncmp(name, 'trans', 5)
                rel(end).name = name(1:5);
                v = fscanf(fid, '%f, ');
                rel(end).v = v';
            elseif strncmp(name, 'ref', 3)
                rel(end).name = name(1:3);
                v = fscanf(fid, '%f, ');
                rel(end).v = v';

                tag = 0;
                if (abs(v(1)) > abs(v(2))) && (abs(v(1)) > abs(v(3)))
                    if v(1) < 0
                        tag = 1;
                    end
                elseif (abs(v(2)) > abs(v(1))) && (abs(v(2)) > abs(v(3)))
                    if v(2) < 0
                        tag = 1;
                    end
                elseif (abs(v(3)) > abs(v(1))) && (abs(v(3)) > abs(v(2)))
                    if v(3) < 0
                        tag = 1;
                    end
                end
                if tag == 1
                    v = -v;
                end

            elseif strncmp(name, 'rot', 3)
                rel(end).name = name(1:3);
                v = fscanf(fid, '%f, ');
                rel(end).v = v';
            else
                disp(['Unknown relation "' name(1:length(name)-1) '"']);
            end
        else
            disp(['Unknown character "' c '"']);
        end
        c = fscanf(fid, '%c', 1);
        c = fscanf(fid, '%c', 1);
    end

    fclose(fid);
end
