clear;clc;
% 更改gTruth的DataSource，也就是原图路径
currentPathDataSource = "E:\try\try";
newPathDataSource = "E:\try\new";
alternativePaths = {[currentPathDataSource newPathDataSource]};
unresolvedPaths = changeFilePaths(gTruth,alternativePaths);

% 更改gTruth的DataSource的LabelData，也就是标签路径
currentPathPixels = "E:\matlab_code\play_play\PixelLabelData";
newPathPixels = "E:\try\new\example_label\PixelLabelData";
alternativePaths = {[currentPathPixels newPathPixels]};
unresolvedPaths = changeFilePaths(gTruth,alternativePaths);
