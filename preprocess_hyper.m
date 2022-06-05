clear;clc;
%% 读取图像(最原始的图像)
% 要自己转成0-255
img_path = 'E:\data\xionganimg\processing\XiongAn_ori.tif';
img = imread(img_path);
[h, w, c] = size(img);

%% 转到0-255
new_img = uint8(zeros(h, w, c));
for i=1:1:c
    each_channel = img(:,:,i);
    each_channel = prepro(each_channel);
    new_img(:,:,i) = each_channel;
end
%% 保存
save_path = 'E:\data\xionganimg\processing\XiongAn.tif';
t = Tiff(save_path, 'w');   % 创建tiff文件
tagstruct.ImageLength = h;
tagstruct.ImageWidth = w;
tagstruct.Photometric = 1;  % 颜色空间解释方式
tagstruct.BitsPerSample = 8;    % 每一个像素的数值位数，这里转换为unit8，因此为8位
tagstruct.SamplesPerPixel = c;  % 每一个像素的波段个数，通常图像为1或3，可是对于遥感影像存在多个波段因此经常大于3
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Software = 'MATLAB';  % 表示生成影像的软件
tagstruct.SampleFormat = 1;     % 表示对数据类型的解释
t.setTag(tagstruct);            % 设置Tiff对象的tag
t.write(new_img);
t.close;

%%
function img_a = prepro(img)
rate = 3;
img = double(img);
img(img > rate * mean(img(:))) = rate * mean(img(:));

% if max(img(:)) == 0
%     temp_max = 1;
% else
%     temp_max = max(img(:));
% end

img_a = uint8(img / max(img(:)) * 255);
end
