import os
import glob
import cv2
import numpy as np
from skimage.measure import label, regionprops

detect_list = glob.glob('images/img*.png')

label_list = glob.glob('images/threshi*.png')
F1 = 0


for detect_img, label_img in zip(detect_list, label_list):

    detect = cv2.imread(detect_img, 0)
    #detect = detect [800:,250: 600]


    #detect = detect[:, 960:960*2]
    #detect = cv2.resize(detect, (1920, 1080))

    # ret, detect_thresh = cv2.threshold(detect, 10, 255, cv2.THRESH_BINARY)

    # detect_thresh = detect_thresh[:, 500:1000]


    label_i = cv2.imread(label_img, 0)
    #label_i = label_i [800:,250: 600]
    # la = label.copy()

    #label = cv2.resize(label, (1920, 1080))
    # ret, detect_label = cv2.threshold(label, 10, 255, cv2.THRESH_BINARY)
    # detect_label = detect_label[:, 500:1000]

    # label_image = label(label_i)

    # for region in regionprops(label_image):
    #     firstpoint = region.coords[0]
    #     index = label_image[firstpoint[0], firstpoint[1]]
    #     if region.area <= 200:
    #         label_image[label_image == index] = 0

    # label_image = np.uint8(label_image)
    # label_image[label_image>0] = 255

    # print(label.shape)

    combine = np.vstack((detect, label_i))
    cv2.imwrite('b.jpg', combine)


    # cv2.imshow('d', label)
    # cv2.waitKey(100)
    tp = float(len(np.where((detect==255)&(label_i==255))[0]))
    tn = float(len(np.where((detect==0)&(label_i==0))[0]))
    fp = float(len(np.where((detect==255)&(label_i==0))[0]))
    fn = float(len(np.where((detect==0)&(label_i==255))[0]))
    print(tp, tn, fp, fn)
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))
    print('name: ', detect_img,'F1:', f1, 'Recall:', recall, 'Precision:', precision)

    F1 += f1
aver = F1/len(detect_list)
print(aver)

    # # print(f1, accuracy, precision, recall)
    # TP = 0
    # TN = 0
    # FN = 0
    # FP = 0
    # input_h, input_w = detect.shape[:2]
    # for i in range(input_h):
    #     for j in range(input_w):
    #         if detect_label[i][j] == 255 and detect_thresh[i][j] ==255:
    #             TP += 1
    #         elif detect_label[i][j] == 0 and detect_thresh[i][j] == 0:
    #             TN += 1
    #         elif detect_label[i][j] == 0 and detect_thresh[i][j] == 255:
    #             FP += 1
    #         elif detect_label[i][j] == 255 and detect_thresh[i][j] == 0:
    #             FN += 1
    # print(TP, TN, FP, FN)




    # if (TP+FN)==0:
    #     Recall = 0
    # else:
    #     Recall = 1.0*TP/(TP+FN)

    # if (TP+FP) == 0:
    #     Precision = 0
    # else:
    #     Precision = 1.0*TP/(TP+FP)

    # if (Recall+Precision) == 0:
    #     F1 = 0
    # else: 
    #     F1 = 2.0*(Recall*Precision)/(recall+Precision)

    # print('F1:', F1, 'Recall:', Recall, 'Precision:', Precision)

