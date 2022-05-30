import os, glob
import cv2

folders = glob.glob('rasp-data/correct/*')
folders = [i for i in folders if os.path.isdir(i)]
#1440*1920
h_sections = 4
w_sections = 4

for i in folders:
	images = sorted(glob.glob(os.path.join(i, '*.jpg')))
	for image_path in images:
		label_path = image_path[:-3]+'txt'
		print(image_path, label_path)
		img = cv2.imread(image_path)
		h, w = img.shape[:2]

		hh = h//h_sections
		ww = w/w_sections

		for h_section in range(h_sections):
			for w_section in range(w_sections):
				print(h_section, w_section)

				crop = img[hh*h_section:hh*(h_section+1), ww*w_section:ww*(w_section+1), :]
				cv2.imwrite('test_{}_{}.jpg'.format(h_section, w_section), crop)
		break
	break
