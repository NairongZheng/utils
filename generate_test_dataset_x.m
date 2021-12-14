clear;clc;
%%
path='D:\项目\data\dataset\raw\yushu\ys_sxz.tif';
image=imread(path);
points=[[6150., 1102.]; [690., 2445.]; [2718., 1306.]];
save_path='D:\项目\data\dataset\raw\yushu\dataset\train\ys_sxz.tif';
points_shape=size(points);
points_num=points_shape(1);
region_size=1024;

%%
scale_rate=1/0.2;
new_points=points*scale_rate;
new_region_size=region_size*scale_rate;
%%
for i=1:points_num
    point=new_points(i,:);
    image(point(2):point(2)+new_region_size, point(1):point(1)+new_region_size,:)=255;
end
imwrite(image, save_path)
