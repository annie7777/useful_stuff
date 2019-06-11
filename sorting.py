import os
import glob
import cv2
import numpy as np

def split(x):
	newname = x.split('_')[0]
	return newname[3:]

class Mydata:
	def __init__(self, root_path):
		#self.rgblist = [x for x in glob.glob(os.path.join(root_path, '*_RGB.jpg'))]
		imagelist = glob.glob(os.path.join(root_path, '*_RGB.jpg'))
		newlist = []
		for x in imagelist:
			x = os.path.basename(x)
			newlist.append(x)
		new = sorted(newlist, key=lambda x:int(split(x)))   
		self.rgblist = new
		self.root_path = root_path

	def __getitem__(self, index):
		img= cv2.imread(os.path.join(self.root_path, self.rgblist[index]))
		print(self.rgblist[index])
		return img

data = Mydata('60_20181026_2137_011_50B/Images')

for i, num in enumerate(data):
	cv2.imshow('test',num)
	cv2.waitKey(100)

