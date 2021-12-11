
clear;clc;
%% 读取黑边图像和原图
image_path = 'D:\项目\data\dataset\raw\yushu\try\ys_sxz.tif';
image = imread(image_path);

egde_path = 'D:\项目\data\dataset\raw\yushu\try\ys_edge.png';
edge = imread(egde_path);

%% 把255的部分变成1
edge_1 = edge == 255;
edge_1 = edge_1 * 1;
edge_1 = uint8(edge_1);

%%
image(:,:,1) = image(:,:,1) .* edge_1;
image(:,:,2) = image(:,:,2) .* edge_1;
image(:,:,3) = image(:,:,3) .* edge_1;

%% 写图像
save_path = 'D:\项目\data\dataset\raw\yushu\try\111.tif';
imwrite(image, save_path);
