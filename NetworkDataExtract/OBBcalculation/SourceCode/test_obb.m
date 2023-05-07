clc;
clear;

addpath(genpath('F:\obb'));

[V,F3] = loadawobj('F:\obb\cylinder_2_0.obj');

m.faces = F3';
m.vertices = V';

obb = part_obb(m);

csv_m = [[obb.center' 0]; [obb.origin' 0]; obb.axes']

csvwrite('obb.csv', csv_m);

% save('drawer_1_box', 'obb')

% load drawer_1_box