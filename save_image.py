import glob
import os
import numpy as np
import cv2

if not os.path.exists('images'):
	os.mkdir('images')
if not os.path.exists('masks'):
	os.mkdir('masks')

datasets = glob.glob(('flower_gt_synth/*'))
print(datasets)

for dataset in datasets:
	dataset_basename = os.path.basename(dataset)

	image_folders = glob.glob(os.path.join(dataset, '*'))
	print(image_folders)

	for image_folder in image_folders:
		print(image_folder)

		image_path = glob.glob((os.path.join(image_folder, 'ImagewithDepth', '*.jpg')))
		print(image_path)
		img_basename = os.path.basename(image_path[0])
		label_path = os.path.join(image_folder, 'PixelLabelData', 'Label_1.png')
		print(label_path)

		img = cv2.imread(image_path[0])
		print(img.shape)
		label = cv2.imread(label_path)
		print(np.unique(label))

		image_newname = dataset_basename+'_'+img_basename[:-3] + 'png'
		print(image_newname)

		cv2.imwrite(os.path.join('images',image_newname), img)
		cv2.imwrite(os.path.join('masks',image_newname), label)

		# label = cv2.imread()
