clear;
% path = "H:\data\遮挡\hide\result\HV_outputs\";
% path = "H:\data\遮挡\增强\A320\png\C_rename_HV\";
% path = "H:\data\遮挡\增强\A320\001resultset\001HV_inputs\";
% path = "H:\data\11sar\result\simulate_theta135\J-20\bp0.1\HH\";
path = 'D:\11\plane_angle\output';
list = dir(path);

% filename = "HV_outputs.gif";
% filename = "C_rename_HV.gif";
% filename = "001HV_input.gif";
% filename = "J-20-HH0.1.gif";
filename = 'Boeing787-10.gif';
for i = 3:length(list)
    a=imread(strcat(path,"\",list(i).name));
    [b,map]=rgb2ind(a,256);
if i ==3
    imwrite(b,map,filename,'gif','LoopCount',Inf,'DelayTime',0.01);
else
    imwrite(b,map,filename,'gif','WriteMode','append','DelayTime',0.01);
end
    
end


% WriterObj = VideoWriter('wmw');
% WriterObj.FrameRate = 100;
% open(WriterObj);
% for i = 3:length(list)
%     frame=imread(strcat(path,"\",list(i).name));
%     writeVideo(WriterObj,frame); 
% end
% 
% close(WriterObj);

