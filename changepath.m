clear all
clc
folder = pwd
foldername = 'img10_Middle_RGB'

gTruthFolder = fullfile(folder,foldername, 'gTruth.mat');
load(gTruthFolder)
oldPathDataSource = '/home/tom/AppleProjectData/apple-data/16_20181019_1548_037_14A14B/Images/img21_idx10_Bottom_DRGB.jpg';
newPathDataSource = fullfile(folder,foldername);

oldPathPixelLabel = '/home/tom/AppleProjectData/apple-data/16_20181019_1548_037_14A14B/Images/img21_idx10_Bottom_DRGB.jpg';
newPathPixelLabel = fullfile(folder,foldername);
alterPaths = {[oldPathDataSource newPathDataSource];[oldPathPixelLabel newPathPixelLabel]};
unresolvedPaths = changeFilePaths(gTruth,alterPaths)
