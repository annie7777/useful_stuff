import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import matplotlib.mlab as mlab
from skimage import io, color
from skimage import feature
import cv2
import csv

# import the necessary packages
from sklearn.cluster import MiniBatchKMeans
import argparse
import glob
import os
import math


root_folder = 'row_trees'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

folder_2 = 'row_all_kmeans'

if not os.path.exists(folder_2):
    os.makedirs(folder_2)

# Kmeans 
def Classfier_Kmeans(n, image, basename):
    # BGR to HSV or LAB

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    h, w = image.shape[:2]
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    nclusters = n
    clt = KMeans(n_clusters = nclusters, random_state = 5)
    labels = clt.fit_predict(image)
    labels = labels.reshape((h, w, 1))
    #print("labels:", labels)
    rgb_label = np.tile(labels, [1,1,3])
    #print("rgb_labels:", rgb_label)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    # convert from L*a*b* to RGB

    quant = cv2.cvtColor(quant, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # display the images and wait for a keypress
    #plot different clusters
    label_image = rgb_label
    component = []
    numberofwhitepixels = []
    for j in range(0,n):
        current_component = np.zeros(image.shape,np.uint8)
        current_component[label_image==j] = image[label_image==j]
        component.append(current_component)
        gray = cv2.cvtColor(current_component, cv2.COLOR_BGR2GRAY)
        numberofwhitepixels.append(np.sum(current_component[:,:,0] > 10))

    print(numberofwhitepixels)

    # index = sorted(numberofwhitepixels)[-2]

    # indices = np.argsort(numberofwhitepixels, axis=None)     
    # aaa = np.where(indices == 2)
    # # print(int(aaa[0]))
    index = np.argmin(numberofwhitepixels)
    # print(indices)
    
    #cv2.imshow("component", component[1])
    #cv2.waitKey(1001)
    component_show = component[0]
    for j in range(1,n):
        component_show = np.vstack((component_show,component[j]))
    
    cv2.imwrite(os.path.join(folder_2, '{}'.format(basename)) ,component_show)
    
    print("Kmeans Done!")
    return component[index]


# Find metadata for rows
###########################Get data from csv####################################
with open(r'gps_utm_interpolation.csv','r') as gps:
    data = list(csv.reader(gps))

data = np.array(data)
data_frame = data[:,0]
data_speed = data[:,1]
data_utm_easting = data[:,2]
data_utm_northing = data[:,3]
############################Get possible row###################################
previous_speed = -1
row_num = []
for index,frame in enumerate(data_speed):
    if previous_speed == '0' and data_speed[index] != '0':
        # print(data_frame[index])
        row_num.append(index)
    elif previous_speed != '0' and data_speed[index] == '0':
        # print(data_frame[index])
        row_num.append(index)
    previous_speed = data_speed[index]
# print(row_num)
###########################Filter false postive row####################################
valid_row = []
previous_index = -1
row_num.pop(0)
for i, index in enumerate(row_num):
    if i < 1:
        previous_index = index
        continue
    if i%2 == 1 and (index - previous_index >1000):
        valid_row.append([previous_index, index])
        # print(i)
    previous_index = index
print('valid row number: ', len(valid_row), ' and they are: ', valid_row)
###########################Get heading####################################
heading = [ ]
for index, row in enumerate(valid_row):
    start_index = row[0]
    end_index = row[1]
    start_easting = float(data_utm_easting[start_index])
    start_northing = float(data_utm_northing[start_index])
    end_easting = float(data_utm_easting[end_index])
    end_northing = float(data_utm_northing[end_index])
    direction = math.atan2((end_northing - start_northing), (end_easting - start_easting))
    direction_degrees = direction * 180/ 3.14
    # print(direction_degrees)
    if direction_degrees < 0:
        heading.append('NtoS')
        print('Row{} heading: '.format(index+1), 'N to S')
        for frame in range(start_index, end_index):
            basename = data_frame[frame]
            print(type(basename))
            if frame < 10000:
                frame_folder = basename[5]
                print(type(frame_folder))
            else:
                frame_folder = basename[5:7]
                print(frame_folder)
            folder = 'frame'+'{}'.format(frame_folder)
            basename_str = np.array2string(basename)
            print(type(basename_str))
            img_file = os.path.join('frames_undis_cut', folder, '{}'.format(basename))
            print(img_file)
            img = cv2.imread(img_file)
            # img = cv2.imread(img_file)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,thresh2 = cv2.threshold(img[:,:,0],127,255,cv2.THRESH_BINARY_INV) 
            # dst = Classfier_Kmeans(3, img, basename)

            row_folder = os.path.join(root_folder,'Row{}{}'.format(index+1, 'NtoS'))
            if not os.path.exists(row_folder):
                os.makedirs(row_folder)
            cv2.imwrite(os.path.join(row_folder, basename), thresh2)
            # break
            

            # print(frame, data_frame[frame])
    else:
        heading.append('StoN')
        print('Row{} heading: '.format(index+1), 'S to N')
        for frame in range(start_index, end_index):
            break
            basename = data_frame[frame]
            # if basename[5] == '6' and frame > 6453:
            #     break
            if frame < 10000:
                frame_folder = basename[5]
                print(frame_folder)
            else:
                frame_folder = basename[5:7]
                print(frame_folder)

            folder = 'frame'+'{}'.format(frame_folder)
            basename_str = np.array2string(basename)
            print(type(basename_str))
            img_file = os.path.join('frames_undis_cut', folder, '{}'.format(basename))
            print(img_file)
            img = cv2.imread(img_file)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,thresh2 = cv2.threshold(img[:,:,0],127,255,cv2.THRESH_BINARY_INV)
            # thresh = thresh2[thresh2 == 255]
            # print(len(thresh))
            # print(img_gray.shape)
            # ret2,th2 = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 

            # dst = Classfier_Kmeans(3, img, basename)

            row_folder = os.path.join(root_folder,'Row{}{}'.format(index+1, 'StoN'))
            if not os.path.exists(row_folder):
                os.makedirs(row_folder)
            cv2.imwrite(os.path.join(row_folder, basename), thresh2)
            # break


    # heading.append(direction_degrees) 
# print(heading)



# frames = glob.glob(os.path.join('frames_undis_cut','frame0','*.jpg'))
# # frames.extend(glob.globfloat(os.path.join('frames','frame1','*.jpg')))

# frame_folder = glob.glob(os.path.join('frames_undis_cut','*'))
# frame_folder.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# print(frame_folder)

# for i, folder in enumerate(frame_folder):
#     if i == 0:
#         continue
#     # print(folder)
#     frames.extend(glob.glob(os.path.join(folder,'*.jpg')))

# frames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# print('Number of Frames: {}'.format(len(frames)))


# folder = 'seg_trees'

# if not os.path.exists(folder):
#     os.makedirs(folder)


# folder_2 = 'all_seg_kmeans'

# if not os.path.exists(folder_2):
#     os.makedirs(folder_2)


# # with open('peaks_new.csv', newline='') as csvfile:
# #     data = list(csv.reader(csvfile))


# # filenames_color = glob.glob("undis_cut/*.jpg")
# # print(len(filenames_color))
# # #print(filenames_color)
# # rgb_files = [cv2.imread(img) for img in filenames_color]
# # #print(rgb_files)
# # i = 0

# # if not os.path.isdir('20181025_011_50_D_Kmeans'):
# #     os.mkdir('20181025_011_50_D_Kmeans')

# # if not os.path.isdir('20181025_011_50_D_Kmeans_all'):
# #     os.mkdir('20181025_011_50_D_Kmeans_all')

# def Classfier_Kmeans(n, image, basename):

#     # BGR to HSV or LAB

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     #image[:,:,0] = cv2.equalizeHist(image[:,:,0])
#     h, w = image.shape[:2]
#     image = image.reshape((image.shape[0] * image.shape[1], 3))
#     nclusters = n
#     clt = KMeans(n_clusters = nclusters, random_state = 5)
#     labels = clt.fit_predict(image)
#     labels = labels.reshape((h, w, 1))
#     #print("labels:", labels)
#     rgb_label = np.tile(labels, [1,1,3])
#     #print("rgb_labels:", rgb_label)
#     quant = clt.cluster_centers_.astype("uint8")[labels]
#     # reshape the feature vectors to images
#     quant = quant.reshape((h, w, 3))
#     image = image.reshape((h, w, 3))
#     # convert from L*a*b* to RGB

#     quant = cv2.cvtColor(quant, cv2.COLOR_HSV2BGR)
#     image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


#     # display the images and wait for a keypress

#     #plot different clusters

#     label_image = rgb_label
#     component = []
#     numberofwhitepixels = []
#     for j in range(0,n):
#         current_component = np.zeros(image.shape,np.uint8)
#         current_component[label_image==j] = image[label_image==j]
#         component.append(current_component)
#         gray = cv2.cvtColor(current_component, cv2.COLOR_BGR2GRAY)
#         numberofwhitepixels.append(np.sum(current_component[:,:,0] > 10))

#     print(numberofwhitepixels)

#     # index = sorted(numberofwhitepixels)[-2]

#     # indices = np.argsort(numberofwhitepixels, axis=None)     
#     # aaa = np.where(indices == 2)
#     # # print(int(aaa[0]))
#     index = np.argmin(numberofwhitepixels)
#     # print(indices)
    
#     #cv2.imshow("component", component[1])
#     #cv2.waitKey(1001)
#     component_show = component[0]
#     for j in range(1,n):
#         component_show = np.vstack((component_show,component[j]))
#     cv2.imwrite(os.path.join(folder_2, basename),component_show)
    
#     print("Kmeans Done!")
#     return component[index]


# for index, f in enumerate(frames):
#     if index < 6462:
#         continue
#     # if i10ndex > :
#     #     break
#     img = cv2.imread(f)
#     basename = os.path.basename(f)
#     dst = Classfier_Kmeans(3, img, basename)
#     cv2.imwrite(os.path.join(folder, basename), dst)

# # # for img in rgb_files:
# # img = cv2.imread('Frame_undistorted/frame5000.jpg')
# # dst = Classfier_Kmeans(3,img,0)
# # cv2.imwrite("20181025_011_50_D_Kmeans/{}".format(basename), dst)
# #     # i +=1
    


	
