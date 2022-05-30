clear all;
clc;

myfolder = pwd;
Images = fullfile(myfolder, 'FlowerImages', '*.JPG')
ImageFiles = dir(Images);

Masks = fullfile(myfolder, 'AppleALabels_Train/Masks_Train', '*.png'); 
MaskFiles = dir(Masks);

newSubFolder = sprintf(int2str(10), myfolder);
if ~exist(newSubFolder, 'dir')
    mkdir(newSubFolder);
end

for k = 1: length(MaskFiles)
    basename_mask = MaskFiles(k).name;
    for j = 1:length(ImageFiles)
        basename_image = ImageFiles(j).name;
        if basename_mask(1:3) == basename_image(6:8)
           right_image = basename_image
           break
        end
    end
    
    fullFilename_mask = fullfile(myfolder, 'AppleALabels_Train/Masks_Train',basename_mask); 
    fullFilename_image = fullfile(myfolder, 'FlowerImages', right_image);
    
    mask = imread(fullFilename_mask);
    image = imread(fullFilename_image);
    small_image = imresize(image, 0.1);
    small_mask = imresize(mask, 0.1);
   
    newSubSubFolder = sprintf(int2str(100),'datasets', myfolder);
    if ~exist(newSubSubFolder, 'dir')
        mkdir(newSubSubFolder);
        mkdir(newSubSubFolder,'images');
        mkdir(newSubSubFolder,'masks');
    end
%     fullfile_i = fullfile(newSubSubFolder, 'images');
%     imwrite(small_image, ['newSubSubFolder', num2str(k), '.png']);
   
    
    

    
    
    %basename_image = ImageFiles(k).name(6:8)

end



dir('~/AppleA/AppleALabels_Train/*');
for k=1:length(Files)
   FileNames=Files(k).name
end


I = imread('02.jpg');
I = imresize(I, 0.3);
imshow(I)
imwrite(I, '002.png')

I1 = imread('002.jpg');
I1 = imresize(I1, 0.3);

bw = imbinarize(I1);
imshow(bw)

cc = bwconncomp(bw,4)

cc.NumObjects
for i=1:cc.NumObjects
    grain = false(size(bw));
    grain(cc.PixelIdxList{i}) = true;
    %imshow(grain)
    imwrite(grain,sprintf('%d.png', i));
end
