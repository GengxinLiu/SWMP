

ºúÈðÕä 2016/9/4 8:59:54
Instructions:

1) Run init.m
2) obb = part_obb(mesh);  -- where mesh is a struct with 'faces' and 'vertices' fields.
3) plot_box(obb, <show_axis>, <paint_box>); -- where <show_axis> = 1 draws the three box axes and <paint_box> = 1 paints the faces of the box (otherwise only the box edges are shown).

You may have to go into SymOOBB and run 'make'. If you run it in Unix it should probably work as is since I just compiled it. It might also work in Windows.