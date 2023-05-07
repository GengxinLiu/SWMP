clc;
clear;

% calcu all mesh obb on the path dir '..\TempData\MatlabInput', and save to
% xxx_box.csv file on the path dir  '..\TempData\PartsOBB'

addpath('./SourceCode/');
addpath('./SourceCode/SymOOBB/');
addpath('./SourceCode/symmetry/');
addpath('./SourceCode/symmetry/Intersection/');

fileFolder=fullfile('./Input');

dirOutput=dir(fullfile(fileFolder,'*.obj'));
fileNames={dirOutput.name}';

for i = 1 : size(fileNames, 1)
    obj_file = ['./Input/' fileNames{i}];
    box_file = ['./Output/', fileNames{i}(1 : end - 4), '_box.csv'];
    
    if exist(box_file, 'file')
        continue;
    end
    
    disp(['\nComputing OBB for ',  fileNames{i}, '...']);
    
    [V,F3] = loadawobj(obj_file);
    
    m.faces = F3';
    m.vertices = V';
    
    obb = part_obb(m);
    
    clf;
    % show_mesh(m, 1, 1);
    % plot_box(obb, 0);
    % saveas(gcf, ['./Output/', fileNames{i}(1 : end - 4), '_box.png'], 'png')
    
    csv_m = [[obb.center' 0]; [obb.origin' 0]; obb.axes'];
    csvwrite(box_file, csv_m);
end
