import os
import sys
sys.path.append("lib/")
import csv
import cv2
import numpy as np

def box_intersection(a, b):
    """
    Intersection area between two rectangles
    Rectangle format: [x1, y1, x2, y2]
    """
    x1 = max(a[0], b[0])
    x2 = min(a[2], b[2])
    y1 = max(a[1], b[1])
    y2 = min(a[3], b[3])
    w, h = x2-x1+1, y2-y1+1
    if w < 0 or h < 0:
        return 0
    else:
        return w*h

def box_union(a, b):
    """
    Union area between two rectangles
    Rectangle format: [x1, y1, x2, y2]
    """
    wa, ha = a[2]-a[0]+1, a[3]-a[1]+1
    wb, hb = b[2]-b[0]+1, b[3]-b[1]+1
    return wa*ha + wb*hb - box_intersection(a,b)

def box_iou(a,b):
    """
    Intersection over Union between two rectangles
    Rectangle format: [x1, y1, x2, y2]
    """
    wa, ha = a[2]-a[0]+1, a[3]-a[1]+1
    wb, hb = b[2]-b[0]+1, b[3]-b[1]+1
    bi = box_intersection(a,b)
    return bi/(wa*ha + wb*hb - bi)

def boxes_intersect(A,b):
    """
    Run intersection with input array of multiple boxes represented by [Nx4] and a single box [4,]
    Output is len N
    """
    #print(A, b)
    ixmin = np.maximum(A[:, 0], b[0])
    iymin = np.maximum(A[:, 1], b[1])
    ixmax = np.minimum(A[:, 2], b[2])
    iymax = np.minimum(A[:, 3], b[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    return inters

def boxes_union(A,b):
    """
    Run union with input array of multiple boxes represented by[Nx4] and a single box[4, ]
    Output is len N
    """
    inters = boxes_intersect(A,b)
    uni = ((b[2] - b[0] + 1.) * (b[3] - b[1] + 1.) +
           (A[:, 2] - A[:, 0] + 1.) *
           (A[:, 3] - A[:, 1] + 1.) - inters)
    return uni

def boxes_iou(A,b):
    """
    Run intersection over union with input array of multiple boxes represented by[Nx4] and a single box[4, ]
    Output is len N
    """
    return boxes_intersect(A,b)/boxes_union(A,b)

# PRECISION RECALL F1-SCORES #
def eval_f1pr(true_positives, false_positives, false_negatives):
    """
    Evaluate detection results: f1-score, precision, recall
    """

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives/float(true_positives+false_positives)

    if true_positives+false_negatives == 0:
        recall = 0
    else:
        recall = true_positives/float(true_positives+false_negatives)

    if precision+recall == 0:
        f1score = 0
    else: 
        f1score = 2*(precision*recall)/float(precision+recall)

    return f1score, precision, recall

def suppress_edges(boxes, image_shape, pad_size=15):
    """
    Suppress box data if its close to the edge
    :param boxes: nx4 box data: x1, y1, x2, y2
    :param image_shape: image shape: [dy, dx]
    :param pad_size: no go zones pixel size
    :return: suppress nx4 boxes
    """

    output_boxes = boxes.copy()

    if boxes.shape[0] > 0:
        max_xy = boxes[:, 2:4]
        min_xy = boxes[:, :2]
        max_outer = np.logical_or(max_xy[:, 0] > image_shape[1]- pad_size, max_xy[:, 1] > image_shape[0] - pad_size)
        min_outer = np.logical_or(min_xy[:, 0] < pad_size, min_xy[:, 1] < pad_size)
        outer_boxes = np.where(np.logical_or(max_outer, min_outer))[0]
        output_boxes = np.delete(output_boxes, outer_boxes, axis=0)

    return output_boxes

def eval_iou_tpfpfn(det_boxes, gt_boxes, min_overlap=0.5,
                    det_multiple_association=False, gt_multiple_association=False):
    """
    Evaluate indices of tp, fn and fn based on intersection over union of detection bounding boxes and ground truth bounding boxes
    :param det_boxes: [nx4] array, cols: [x1, y1, x2, y2]
    :param gt_boxes: [kx4] array, cols: [x1, y1, x2, y2]
    :param min_overlap: float - minimum iou for an object to be classified as tp
    :param det_multiple_association: Allows one detection to be associated with multiple GTs
    :param gt_multiple_association: Allows one GT to be associated with multiple detections
    :return: indices for TP, FP and FN.
             The indices for TP and FP are wrt input detections
             The indices for FN are wrt input groundtruth
    """

    # Check if either we have zero detection or zero ground truth
    if det_boxes.shape[0] == 0:
        tp = np.array([])
        fp = np.array([])
        fn = np.arange(gt_boxes.shape[0])
        return tp, fp, fn
    if gt_boxes.shape[0] == 0:
        tp = np.array([])
        fp = np.arange(det_boxes.shape[0])
        fn = np.array([])
        return tp, fp, fn

    # For each gt, evaluate the iou to all other det
    ### USE THIS CODE IF BBOX_OVERLAPS DOES NOT EXIST - its just a little bit slower
    all_res = np.zeros((gt_boxes.shape[0], det_boxes.shape[0]))
    for gti in range(gt_boxes.shape[0]):
        all_res[gti, :] = boxes_iou(det_boxes, gt_boxes[gti, :])
    ### WITH BBOX_OVERLAPS
    # all_res = bbox_overlaps(gt_boxes.astype(np.float), det_boxes.astype(np.float))

    # all_res, cols: detection, rows: gt
    # Each detection can only relate to a single gt, suppress others
    if not det_multiple_association:
        all_res[all_res < all_res.max(axis=0)] = 0
    # Each Ground truth can only be represented by a single detection
    if not gt_multiple_association:
        all_res[all_res < all_res.max(axis=1)[:, None]] = 0

    # Eval tp, fp, fn
    fn = np.where(all_res.max(axis=1) < min_overlap)[0]
    tp = np.where(all_res.max(axis=0) > min_overlap)[0]
    fp = np.where(all_res.max(axis=0) < min_overlap)[0]

    return tp, fp, fn

from infer_detector import Infer
gtf = Infer();
gtf.Model(model_name="vgg", weights="weights/Best_RFB_vgg_COCO.pth", use_gpu=True)
dataset_dir = '../datasets/cherries_fruitlets_crumpet-1'
image_dir = 'JPEGImages'
txt_label_dir = 'labels'
class_file = dataset_dir+"/annotations/classes.txt"
gtf.Image_Params(class_file, input_size=512)
gtf.Setup()
test_csv = 'test.txt'
overlay_dir = 'overlay'



for th in np.arange(0.65, 1, 0.1):
    TP, FP, FN = 0, 0, 0
    with open(dataset_dir+test_csv, 'r') as test_lists:
        for line in csv.reader(test_lists):
#             print(line)
            img_path = os.path.join(dataset_dir, image_dir, os.path.basename(line[0])+'.png')
            img = cv2.imread(img_path)
            dt_boxes = gtf.Predict(img_path, thresh=th, font_size=1, line_size=2)
            dt = []
            for dt_box in dt_boxes:
                pt_1 = (dt_box[0], dt_box[1])
                pt_2 = (dt_box[2], dt_box[3])
                dt.append([dt_box[0],dt_box[1],dt_box[2], dt_box[3]])

                cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 1)
            label_path = os.path.join(dataset_dir, txt_label_dir, os.path.basename(line[0])+'.txt')
            gt_ts = []
            gt_boxes =[]
            with open(label_path, 'r') as gt:
                for bboxes in gt:
                    values = bboxes.split(" ")
                    x, y, w, h = values[1], values[2], values[3], values[4]
                    gt_ts.append(0)
                    xc, yc, ww, hh = int(float(x)*512), int(float(y)*512), int(float(w)*512), int(float(h)*512)
                    xmin = xc - int(ww/2)
                    ymin = yc - int(hh/2)
                    pt1 = (xmin, ymin)
                    pt2 = (xmin+ww, ymin+hh)
                    gt_boxes.append([xmin,ymin, xmin+ww, ymin+hh])
                    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
    #         print(os.path.basename(line[0]))
            cv2.imwrite(os.path.join('overlay',os.path.basename(line[0])+'.png'), img) 

    #         print(dt)
            gt_boxes = np.array(gt_boxes)
            dt = np.array(dt)
            tp, fp, fn = eval_iou_tpfpfn(dt, gt_boxes, min_overlap=0.5,
                                                   det_multiple_association=False, gt_multiple_association=False)
    #         print(len(tp), len(fp), len(fn))
            TP +=len(tp)
            FP +=len(fp)
            FN +=len(fn)


    F1,P,R = eval_f1pr(TP,FP,FN)
    print(th, F1, P, R, TP, FP, FN)

        
#         print(gt_boxes, dt_boxes)
        
# img_path = "../sample_dataset/kangaroo2/Images/20191009T003949.382096.png";
# output = gtf.Predict(img_path, thresh=0.25, font_size=1, line_size=2)
# print(output)

# from IPython.display import Image
# Image(filename='output.png') 
