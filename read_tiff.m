t0=Tiff('Realsense_shifted_fcn_50A_new.tiff','r');
t1=Tiff('Realsense_shifted_fcn_50B_new.tiff','r');


imageData0=read(t0);
imageData1=read(t1);


imageData0=imageData0(:,:);
imageData1=imageData1(:,:);


imageData0 = double(imageData0);
imageData1 = double(imageData1);
% 
% imageData0 = [zeros(8,10) imageData0];
% imageData1 = [imageData1 zeros(8,10)];

M1 = max(imageData0, [], 'all')
M2 = max(imageData1, [], 'all')

imageData0 = imageData0/8513*10;
imageData1 = imageData1/8513*10;

imageDiff = abs(imageData0-imageData1);

M3 = max(imageDiff, [], 'all')

% subplot(3,1,1);
% imshow(imageData0,[0,10]);% specify color range
% colormap copper

% subplot(3,1,2);
% imshow(imageData1,[0,10]);
% colormap copper
% 
% subplot(3,1,3);
imshow(imageDiff,[0,10]);
colormap copper
iptsetpref('ImshowBorder','tight');
saveas(gcf,'densitymaping2018day_50diff.png')

R1=corrcoef(imageData0(:), imageData1(:))
