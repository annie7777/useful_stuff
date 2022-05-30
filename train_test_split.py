from sklearn.model_selection import train_test_split
import os, glob
import argparse
import csv
import time, datetime
import cv2
import pandas as pd 

csv_files = pd.read_csv('quads_pixels_max.csv')

X_train, X_test = train_test_split(csv_files, test_size = 0.33,random_state = 42)
print(X_test, len(X_train))
X_test.to_csv('test.csv', encoding = 'utf-8', index = False)
X_train.to_csv('train.csv', encoding = 'utf-8', index = False)
# parser = argparse.ArgumentParser(description = 'Split Training and tesating dataset')
# parser.add_argument('--rootdir', required = True, metavar = "/path/to/dataset", help = "root directory of the dataset")
# parser.add_argument('--imagedir', required = True, metavar = "/path/to/images", help = "image directory of the dataset")
# parser.add_argument('--labeldir', required = True, metavar = "/path/to/labels", help = "label directory of the dataset")
# args  = parser.parse_args()

# rootdir = args.rootdir
# imagedir = args.imagedir
# labeldir = args.labeldir

# images = glob.glob(os.path.join(rootdir, imagedir, '*.jpg'))
# images.extend(glob.glob(os.path.join(rootdir, imagedir, '*.bmp')))

# # labels = glob.glob(os.path.join(rootdir, labeldir, '*.jpg'))
# # labels.extend(glob.glob(os.path.join(rootdir, labeldir, '*.bmp')))

# X_train, X_test = train_test_split(images, test_size = 0.33,random_state = 42)

# print(len(X_train), len(X_test))

# time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# train_csv = 'train_csv'+ time +'.csv'
# test_csv = 'test_csv'+ time + '.csv'

# def write_csv(data, csv_file):
#     with open(csv_file, 'a', newline = '') as file:
#         writer = csv.writer(file)
#         writer.writerow(data)

# for file in X_train:
#     basename = os.path.basename(file)
#     label_path = os.path.join(rootdir, labeldir, basename)

#     new_row = [ ]
#     new_row.append(file)
#     new_row.append(label_path)

#     write_csv(new_row, train_csv)


# for file in X_test:
#     basename = os.path.basename(file)
#     label_path = os.path.join(rootdir, labeldir, basename)

#     new_row = [ ]
#     new_row.append(file)
#     new_row.append(label_path)

#     write_csv(new_row, test_csv)


