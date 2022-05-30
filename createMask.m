%% One class
clear all
I=0;
load(strcat('img_',num2str(I),'_idx_None_imx12_middle_rgb\gTruth.mat'))
label1 = gTruth.LabelData.GT_HG{1,1};
im = imread(strcat('img_',num2str(I),'_idx_None_imx12_middle_rgb\ImagewithDepth\img_',...
                    num2str(I),'_idx_None_imx12_middle_rgb.jpg'));
%subplot(2,1,1);
% imshow(im)
msk=zeros(size(im,1),size(im,2));
for i=1:size(label1,1)
x1 = label1(i,1);
x2 = x1+label1(i,3);
y1 = label1(i,2);
y2 = y1+label1(i,4);
msk(y1:y2,x1:x2) = 1;
end
%subplot(2,1,2);
% figure
% imshow(msk)
imwrite(uint8(msk)*255,strcat(num2str(I),'.jpg'));

%% Two class
clear all
I=103;
load(strcat('img_',num2str(I),'_idx_None_imx12_middle_rgb\gTruth_2cls.mat'))
label1 = gTruth.LabelData.Tips{1,1};
label2=gTruth.LabelData.Greens{1,1};
im = imread(strcat('img_',num2str(I),'_idx_None_imx12_middle_rgb\ImagewithDepth\img_',...
                    num2str(I),'_idx_None_imx12_middle_rgb.jpg'));

msk=zeros(size(im,1),size(im,2));
for i=1:size(label1,1)
x1 = label1(i,1);
x2 = x1+label1(i,3);
y1 = label1(i,2);
y2 = y1+label1(i,4);
msk(y1:y2,x1:x2) = 1;
end
imwrite(uint8(msk)*255,strcat([num2str(I) '_Tips'],'.jpg'));

msk=zeros(size(im,1),size(im,2));
for i=1:size(label2,1)
x1 = label2(i,1);
x2 = x1+label2(i,3);
y1 = label2(i,2);
y2 = y1+label2(i,4);
msk(y1:y2,x1:x2) = 1;
end
%subplot(2,1,2);
% figure
% imshow(msk)
imwrite(uint8(msk)*255,strcat([num2str(I) '_Greens'],'.jpg'));