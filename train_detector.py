import sys
import os
import csv
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
from layers.functions import Detect
from utils.nms_wrapper import nms
import time
from swift_box_overlay import swift_box_overlay

f1_evaluation = swift_box_overlay()
class Detector():
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["dataset"] = {};
        self.system_dict["dataset"]["train"] = {};
        self.system_dict["dataset"]["val"] = {};
        self.system_dict["dataset"]["val"]["status"] = False;

        self.system_dict["params"] = {};

        self.system_dict["params"]["version"] = "RFB_vgg"; #RFB_E_vgg or RFB_mobile version
        self.system_dict["params"]["basenet"] = "weights/vgg16_reducedfc.pth";
        self.system_dict["params"]["cuda"] = True;
        self.system_dict["params"]["ngpu"] = 1;

        self.system_dict["params"]["dataset"] = "COCO";
        self.system_dict["params"]["num_workers"] = 3;
        self.system_dict["params"]["size"] = 512; #300;
        self.system_dict["params"]["batch_size"] = 4;

        self.system_dict["params"]["jaccard_threshold"] = 0.5;
        self.system_dict["params"]["lr"] = 0.0001;
        self.system_dict["params"]["momentum"] = 0.9;
        self.system_dict["params"]["weight_decay"] = 0.0005;
        self.system_dict["params"]["gamma"] = 0.1;

        self.system_dict["params"]["resume_epoch"] = 0
        self.system_dict["params"]["resume_net"] = None;
        
        self.system_dict["params"]["max_epoch"] = 1000;
        self.system_dict["params"]["log_iters"] = True;
        self.system_dict["params"]["save_folder"] = "weights/";


    def Train_Dataset(self, root_dir, coco_dir, set_dir, batch_size=4, image_size=512, num_workers=3):
        self.system_dict["dataset"]["train"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["train"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["train"]["set_dir"] = set_dir;

        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["size"] = image_size;
        self.system_dict["params"]["num_workers"] = num_workers;


    def Val_Dataset(self, root_dir, coco_dir, set_dir):
        self.system_dict["dataset"]["val"]["status"] = True;
        self.system_dict["dataset"]["val"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["val"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["val"]["set_dir"] = set_dir;  
        

    def Model(self, model_name="vgg", use_gpu=True, ngpu=1):
        if(not os.path.isdir("weights/")):
            cmd1 = "cp " + os.path.dirname(os.path.realpath(__file__)) + "/download.sh " + os.getcwd() + "/.";
            os.system(cmd1);
            os.system("chmod +x download.sh");
            os.system("./download.sh");
        if(model_name == "vgg"):
            self.system_dict["params"]["version"] = "RFB_vgg";
            self.system_dict["params"]["basenet"] = "weights/vgg16_reducedfc.pth";
        elif(model_name == "e_vgg"):
            self.system_dict["params"]["version"] = "RFB_E_vgg";
            self.system_dict["params"]["basenet"] = "weights/vgg16_reducedfc.pth";
        elif(model_name == "mobilenet"):
            self.system_dict["params"]["basenet"] = "weights/mobilenet_feature.pth";
            self.system_dict["params"]["version"] = "RFB_mobile";

        self.system_dict["params"]["cuda"] = use_gpu;
        self.system_dict["params"]["ngpu"] = ngpu;


    def Set_HyperParams(self, lr=0.0001, momentum=0.9, weight_decay=0.0005, gamma=0.1, jaccard_threshold=0.5):
        self.system_dict["params"]["jaccard_threshold"] = jaccard_threshold;
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["momentum"] = momentum;
        self.system_dict["params"]["weight_decay"] = weight_decay;
        self.system_dict["params"]["gamma"] = gamma;
#         self.system_dict["params"]["resume_net"] = 'weights/RFB_vgg_COCO_last.pth'
        print('yes it is the lib inside the example book')
    
    def Image_Params(self, class_file, input_size=512):
        self.system_dict["params"]["size"] = input_size;
        f = open(class_file, 'r');
        lines = f.readlines();
        f.close();
        for i in range(len(lines)):
            self.system_dict["local"]["classes"].append(lines[i][:len(lines[i])-1])
        
        
    def EVAL(self, model, priorbox):
        line_size=1
        font_size=1
        
        dataset_dir = self.system_dict["dataset"]["train"]["root_dir"]+'/'+self.system_dict["dataset"]["train"]["coco_dir"]
        image_dir = self.system_dict["dataset"]["train"]["set_dir"]
        txt_label_dir = 'labels'
        class_file = dataset_dir+'/annotations/classes.txt'
        test_csv = 'test.txt'
        overlay_dir = 'overlay'
        
        if not os.path.exists(os.path.join('overlay')):
            os.mkdir(os.path.join('overlay'))

        num_classes = 2
        
        self.system_dict["local"]["priorbox"] = PriorBox(COCO_512)
        self.system_dict["local"]["classes"] = ["__bg__"];
        self.Image_Params(class_file, input_size=512)
        
        self.system_dict["params"]["cpu"] = False;
        cuda = self.system_dict["params"]["cuda"]

        model = model.eval()
        transform = BaseTransform(model.size, (25, 25, 25), (2, 0, 1))
        
        with torch.no_grad():
            self.system_dict["local"]["priors"] = self.system_dict["local"]["priorbox"].forward()
            if self.system_dict["params"]["cuda"]:
                self.system_dict["local"]["priors"] = self.system_dict["local"]["priors"].cuda()
        
        self.system_dict["local"]["detector"] = Detect(num_classes, 0, COCO_512)


        TP, FP, FN = 0, 0, 0
        cuda = True

        with open(dataset_dir+'/'+test_csv, 'r') as test_lists:
            for line in csv.reader(test_lists):
                img_path = os.path.join(dataset_dir, image_dir, os.path.basename(line[0]))
                img = cv2.imread(img_path)
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                with torch.no_grad():
                    x = transform(img).unsqueeze(0)
                    if cuda:
                        x = x.cuda()
                        scale = scale.cuda()
                        
                out=model(x)
                boxes, scores = self.system_dict["local"]["detector"].forward(out, self.system_dict["local"]["priors"])
                boxes = boxes[0]
                scores=scores[0]

                boxes *= scale
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                # scale each detection back up to the image

                classes = self.system_dict["local"]["classes"];
                all_boxes = []

                for j in range(1, num_classes):
                    inds = np.where(scores[:, j] > 0.5)[0]
                    if len(inds) == 0:
                        continue
                    c_bboxes = boxes[inds]
                    c_scores = scores[inds, j]
                    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                        np.float32, copy=False)

                    keep = nms(c_dets, 0.4, force_cpu=self.system_dict["params"]["cpu"])
                    c_dets = c_dets[keep, :]
                    c_bboxes=c_dets[:, :4]

                    for bbox in c_bboxes:
                        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), line_size)
                        cv2.putText(img, classes[j], (int(bbox[0]), int(bbox[1])) , 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), line_size)
                        all_boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])#, classes[j]])

#                     cv2.imwrite(os.path.basename(line[0]), img)
                label_path = os.path.join(dataset_dir, txt_label_dir, os.path.basename(line[0])[:-4] +'.txt')
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
                cv2.imwrite(os.path.join('overlay',os.path.basename(line[0])), img) 

        #         print(dt)
                gt_boxes = np.array(gt_boxes)
                dt = np.array(all_boxes)
#                 print(gt_boxes, all_boxes)
                tp, fp, fn = f1_evaluation.eval_iou_tpfpfn(dt, gt_boxes, min_overlap=0.5,
                                                       det_multiple_association=True, gt_multiple_association=True)
#                 print(len(tp), len(fp), len(fn))
                TP +=len(tp)
                FP +=len(fp)
                FN +=len(fn)

        F1,P,R = f1_evaluation.eval_f1pr(TP,FP,FN)
        print('test f1, precision, recall, tp, fp, fn: ', F1, P, R, TP, FP, FN)
                
        return F1



    def Train(self, epochs=200, log_iters=True, output_weights_dir="weights", saved_epoch_interval=10):
        self.system_dict["params"]["max_epoch"] = epochs;
        self.system_dict["params"]["log_iters"] = log_iters;
        self.system_dict["params"]["save_folder"] = output_weights_dir;

        if not os.path.exists(self.system_dict["params"]["save_folder"]):
            os.mkdir(self.system_dict["params"]["save_folder"])

        if(self.system_dict["params"]["size"] == 300):
            cfg = COCO_300;
        else:
            cfg = COCO_512;

        if self.system_dict["params"]["version"] == 'RFB_vgg':
            from models.RFB_Net_vgg import build_net
        elif self.system_dict["params"]["version"] == 'RFB_E_vgg':
            from models.RFB_Net_E_vgg import build_net
        elif self.system_dict["params"]["version"] == 'RFB_mobile':
            from models.RFB_Net_mobile import build_net
            cfg = COCO_mobile_300
        else:
            print('Unkown version!')


        
        img_dim = (300,512)[self.system_dict["params"]["size"]==512]
        rgb_means = ((25, 25, 25),(25,25,25))[self.system_dict["params"]["version"] == 'RFB_mobile']
        p = (0.6,0.2)[self.system_dict["params"]["version"] == 'RFB_mobile']
        
        f = open(self.system_dict["dataset"]["train"]["root_dir"] + "/" + 
            self.system_dict["dataset"]["train"]["coco_dir"] + "/annotations/classes.txt", 'r');
        lines = f.readlines();
        if(lines[-1] == ""):
            num_classes = len(lines) - 1;
        else:
            num_classes = len(lines) + 1;

        
        batch_size = self.system_dict["params"]["batch_size"]
        weight_decay = self.system_dict["params"]["weight_decay"]
        gamma = self.system_dict["params"]["gamma"]
        momentum = self.system_dict["params"]["momentum"]

        self.system_dict["local"]["net"] = build_net('train', img_dim, num_classes)

        if self.system_dict["params"]["resume_net"] == None:
            base_weights = torch.load(self.system_dict["params"]["basenet"])
            print('Loading base network...')
            self.system_dict["local"]["net"].base.load_state_dict(base_weights)

            def xavier(param):
                init.xavier_uniform(param)

            def weights_init(m):
                for key in m.state_dict():
                    if key.split('.')[-1] == 'weight':
                        if 'conv' in key:
                            init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                        if 'bn' in key:
                            m.state_dict()[key][...] = 1
                    elif key.split('.')[-1] == 'bias':
                        m.state_dict()[key][...] = 0

            print('Initializing weights...')
        # initialize newly added layers' weights with kaiming_normal method
            self.system_dict["local"]["net"].extras.apply(weights_init)
            self.system_dict["local"]["net"].loc.apply(weights_init)
            self.system_dict["local"]["net"].conf.apply(weights_init)
            self.system_dict["local"]["net"].Norm.apply(weights_init)
            if self.system_dict["params"]["version"] == 'RFB_E_vgg':
                self.system_dict["local"]["net"].reduce.apply(weights_init)
                self.system_dict["local"]["net"].up_reduce.apply(weights_init)

        else:
        # load resume network
            print('Loading resume network: {}'.format(self.system_dict["params"]["resume_net"]))
            state_dict = torch.load(self.system_dict["params"]["resume_net"])
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            self.system_dict["local"]["net"].load_state_dict(new_state_dict)


        if self.system_dict["params"]["ngpu"] > 1:
            self.system_dict["local"]["net"] = torch.nn.DataParallel(self.system_dict["local"]["net"], device_ids=list(range(self.system_dict["params"]["ngpu"])))

        if self.system_dict["params"]["cuda"]:
            self.system_dict["local"]["net"].cuda()
            cudnn.benchmark = True

        
        optimizer = optim.SGD(self.system_dict["local"]["net"].parameters(), lr=self.system_dict["params"]["lr"],
                              momentum=self.system_dict["params"]["momentum"], weight_decay=self.system_dict["params"]["weight_decay"])
        #optimizer = optim.RMSprop(self.system_dict["local"]["net"].parameters(), lr=self.system_dict["params"]["lr"], alpha = 0.9, eps=1e-08,
        #                      momentum=self.system_dict["params"]["momentum"], weight_decay=self.system_dict["params"]["weight_decay"])

        criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
        priorbox = PriorBox(cfg)
#         print(priorbox)
        with torch.no_grad():
            priors = priorbox.forward()
            if self.system_dict["params"]["cuda"]:
                priors = priors.cuda()


        self.system_dict["local"]["net"].train()
        # loss counters
        loc_loss = 0  # epoch
        conf_loss = 0
        epoch = 0 + self.system_dict["params"]["resume_epoch"]
        print('Loading Dataset...')

        if(os.path.isdir("coco_cache")):
            os.system("rm -r coco_cache")

        dataset = COCODetection(self.system_dict["dataset"]["train"]["root_dir"], 
                                self.system_dict["dataset"]["train"]["coco_dir"], 
                                self.system_dict["dataset"]["train"]["set_dir"], 
                                preproc(img_dim, rgb_means, p))


        epoch_size = len(dataset) // self.system_dict["params"]["batch_size"]
        max_iter = self.system_dict["params"]["max_epoch"] * epoch_size

        stepvalues = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
        print('Training', self.system_dict["params"]["version"], 'on', dataset.name)
        step_index = 0

        if self.system_dict["params"]["resume_epoch"] > 0:
            start_iter = self.system_dict["params"]["resume_epoch"] * epoch_size
        else:
            start_iter = 0

        lr = self.system_dict["params"]["lr"]
        previous_f1 = 0


        for iteration in range(start_iter, max_iter):
            if iteration % epoch_size == 0:
                # create batch iterator
                batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                      shuffle=True, num_workers=self.system_dict["params"]["num_workers"], 
                                                      collate_fn=detection_collate))
                loc_loss = 0
                conf_loss = 0
                
                torch.save(self.system_dict["local"]["net"].state_dict(), self.system_dict["params"]["save_folder"] + "/" + self.system_dict["params"]["version"]+'_'+
                               self.system_dict["params"]["dataset"] + '_epoches_'+
                               repr(epoch) + '.pth')
                torch.save(self.system_dict["local"]["net"].state_dict(), self.system_dict["params"]["save_folder"] + "/" + self.system_dict["params"]["version"]+'_'+
                               self.system_dict["params"]["dataset"] + '_epoches_'+
                               'last' + '.pth')
                epoch += 1

            load_t0 = time.time()
            if iteration in stepvalues:
                step_index += 1
            lr = self.adjust_learning_rate(optimizer, self.system_dict["params"]["gamma"], epoch, step_index, iteration, epoch_size)


            # load train data
            images, targets = next(batch_iterator)

            #print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

            if self.system_dict["params"]["cuda"]:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda()) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno) for anno in targets]
            # forward
            t0 = time.time()
            out = self.system_dict["local"]["net"](images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, priors, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            load_t1 = time.time()
  

            if iteration % saved_epoch_interval == 0:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                      + '|| Current iter ' +
                      repr(iteration) + '|| Total iter ' + repr(max_iter) + 
                      ' || L: %.4f C: %.4f||' % (
                    loss_l.item(),loss_c.item()) + 
                    'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
                f1 = self.EVAL(self.system_dict["local"]["net"], priorbox)
                self.system_dict["local"]["net"].train()
                if previous_f1 < f1:
                    torch.save(self.system_dict["local"]["net"].state_dict(), self.system_dict["params"]["save_folder"] + "/" +
                   'Best_' + self.system_dict["params"]["version"] +'_' + self.system_dict["params"]["dataset"] + '.pth')
                    previous_f1 = f1

        torch.save(self.system_dict["local"]["net"].state_dict(), self.system_dict["params"]["save_folder"] + "/" +
                   'Final_' + self.system_dict["params"]["version"] +'_' + self.system_dict["params"]["dataset"] + '.pth')

        



    def adjust_learning_rate(self, optimizer, gamma, epoch, step_index, iteration, epoch_size):
        if epoch < 6:
            lr = 1e-6 + (self.system_dict["params"]["lr"]-1e-6) * iteration / (epoch_size * 5) 
        else:
            lr = self.system_dict["params"]["lr"] * (gamma ** (step_index))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr