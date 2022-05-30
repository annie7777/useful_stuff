import os
import sys
sys.path.append("lib/")
from train_detector import Detector

gtf = Detector();

# #USing training set structured in format 1
# root_dir = "Monk_Object_Detection/example_notebooks/sample_dataset";
# coco_dir = "kangaroo";
# set_dir = "Images";

root_dir = "../datasets";
coco_dir = "cherries_fruitlets_crumpet-1";
set_dir = "JPEGImages";
#300
#512
gtf.Train_Dataset(root_dir, coco_dir, set_dir, batch_size=8,image_size=512, num_workers=3)

#vgg - 300, 512
#e_vgg - 300, 512
#mobilenet - 300, 512
gtf.Model(model_name="vgg", use_gpu=True, ngpu=1)
gtf.Set_HyperParams(lr=0.0001, momentum=0.9, weight_decay=0.0005, gamma=0.1, jaccard_threshold=0.5)
gtf.Train(epochs=1000, log_iters=True, output_weights_dir="weights", saved_epoch_interval=10)