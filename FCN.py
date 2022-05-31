# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from BagData import dataloader, input_h, input_w
from bagdata_test import dataloader_test
import pdb
import numpy as np 
import time
import visdom
import numpy as np
import cv2

print(input_h, input_w)
#input_h = 1280
#input_w = 1920


class FCN32s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN16s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p = 0.5)
        self.dropout2 = nn.Dropout(p = 0.6)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = self.dropout2(score)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.dropout2(score)
        score = score + x3                               # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.dropout2(score)
        score = score + x2                             # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.dropout2(score)
        score = score + x1                    # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.dropout2(score)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def check_accuracy(fcn_model):
    ii = 0
    F1 = 0
    f1_accuracy = []
    TP_test = 0
    TN_test = 0
    FP_test = 0
    FN_test = 0
    
    for item in dataloader_test:
        input_test = item['A']
        origin = item['B']
        name = item['C']

        input_test = torch.autograd.Variable(torch.FloatTensor(input_test))
        input_test = input_test.cuda()
    
        output_test = fcn_model(input_test)

        output_test_np = output_test.cpu().data.numpy().copy()
        output_test_np = np.argmin(output_test_np, axis=1)
#         print(np.unique(output_test_np))
        #print(output_np)
        output_test = np.uint8(output_test_np*255)
        #print(output_test.shape)
        output_test = output_test.reshape((input_h,input_w))
        
        #print(output_test.shape)
        #output_test = cv2.resize(output_test, (1920, 1080))
        #kernel = np.ones((5,5),np.uint8)
        #opening = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
        #print(output.shape)
        name = np.array(name)
        #print(name)
        imgB = cv2.imread('testing_label/'+name[0], 0)
        imgB = imgB[300:1000,500:1500]
#         print(np.unique(imgB))
        output_test = cv2.resize(output_test, (imgB.shape[1], imgB.shape[0]))
        ret, output_test_t = cv2.threshold(output_test,1, 255, cv2.THRESH_BINARY) 
        
#         print(np.unique(output_test_t))
        #imgB = cv2.resize(imgB,(input_w,input_h))
#         print(imgB.shape)
        cv2.imwrite('test_results/'+name[0],output_test_t)    
#         ret, threshold = cv2.threshold(imgB,10, 255, cv2.THRESH_BINARY) 
        #print(np.unique(threshold))
        
        cv2.imwrite('test_results/'+('thresh'+name[0]),imgB)    
#         TP = 0
#         TN = 0
#         FN = 0
#         FP = 0
#         for i in range(input_h):
#             for j in range(input_w):
#                 if threshold[i][j] == 255 and output_test[i][j] ==255:
#                     TP += 1
#                 elif threshold[i][j] == 0 and output_test[i][j] == 0:
#                     TN += 1
#                 elif threshold[i][j] == 0 and output_test[i][j] == 255:
#                     FP += 1
#                 elif threshold[i][j] == 255 and output_test[i][j] == 0:
#                     FN += 1
        #print(TP, TN, FN, FP)
        tp = float(len(np.where((output_test_t==255)&(imgB==255))[0]))
        tn = float(len(np.where((output_test_t==0)&(imgB==0))[0]))
        fp = float(len(np.where((output_test_t==255)&(imgB==0))[0]))
        fn = float(len(np.where((output_test_t==0)&(imgB==255))[0]))
        
        TP_test += tp
        TN_test += tn
        FP_test += fp
        FN_test += fn
    
    if TP_test + TN_test + FP_test + FN_test == (output_test_t.shape[0]*output_test_t.shape[1]* 27):
        if TP_test+TN_test+FP_test+FN_test == 0:
            accuracy = 0
        else: 
            accuray = (TP_test + TN_test)/(TP_test+TN_test+FP_test+FN_test)
        if TP_test+FP_test == 0:
            precision = 0
        else:         
            precision = (TP_test)/(TP_test+FP_test)
        if TP_test+FN_test == 0:
            recall = 0
        else:
            recall = TP_test/(TP_test+FN_test)
    else: 
        print(output_test_t.shape[0], output_test_t.shape[1], len(dataloader_test) )
        print('wrong test evaluation')
    
    if precision+recall ==0:
        F1 = 0
    else:
        F1 = 2*((precision*recall)/(precision+recall))

    print('Average F1 score: {}'.format(F1))
    return F1


if __name__ == "__main__":

    vis = visdom.Visdom()
    vis.close()
    #global plotter
    #plotter = utils.VisdomLinePlotter(env_name='Tutorial Plots')
    #torch.cuda.empty_cache() 
    vgg_model = VGGNet(requires_grad=True)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    #fcn_model = torch.load('checkpoints/fcn_model_40.pt')
    fcn_model = fcn_model.cuda()
    pos_weight = torch.Tensor([(100 / 100)])
    #criterion = nn.BCELoss().cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-1, momentum=0.95)
    #input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    #y = torch.autograd.Variable(torch.randn(batch_size, n_class, h, w), requires_grad=False)
    saving_index =0
    #epoch_loss= [1,2,3,4]
    vis.line(X = np.array([0]), Y = np.array([0]), win = 'f1-score' , name='training',opts=dict(title='f1-score'),)
    vis.line(X = np.array([0]), Y = np.array([0]), win = 'f1-score' , name='testing',opts=dict(title='f1-score'),)
    vis.line(X = np.array([0]), Y = np.array([0/2]), win = 'loss' , name='line1',opts=dict(title='loss'),)
    
    #F1 = check_accuracy(fcn_model)
    #vis.line(X = np.array([1]), Y = np.array([F1]), win = 'f1-score' , name='line1',update='append',)
    for epo in range(100):
        TP_train = 0
        FP_train = 0
        TN_train = 0
        FN_train = 0
        saving_index +=1
        index = 0
        epo_loss = 0
        start = time.time()
        F1_training= 0
        for item in dataloader:
            index += 1
            start = time.time()
            input = item['A']
            y = item['B']
            name = item['C']
            input = torch.autograd.Variable(input)
            y = torch.autograd.Variable(y)
            #fcn_model = fcn_model.cuda()

            input = input.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            output = fcn_model(input)
            #x55 = torch.sum(x5, 0)
            #x55 = x5.cpu().data.numpy().copy()
            #x55 = x55.reshape((512,40,60))
            #x555 = np.sum(x55, axis=0)
            #print(x555.shape)
            #x555 = cv2.resize(x555, (1920, 1080))
            #output = nn.functional.sigmoid(output)
            loss = criterion(output, y)
            loss.backward()
            iter_loss = loss.data 
            epo_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().data.numpy().copy()
            #print(output_np)
            output_np = np.argmin(output_np, axis=1)
            
            y_np = y.cpu().data.numpy().copy()
            y_np = np.argmin(y_np, axis=1)
            
            tp = float(len(np.where((output_np==1)&(y_np==1))[0]))
            tn = float(len(np.where((output_np==0)&(y_np==0))[0]))
            fp = float(len(np.where((output_np==1)&(y_np==0))[0]))
            fn = float(len(np.where((output_np==0)&(y_np==1))[0]))
            
            TP_train += tp
            FP_train += fp
            TN_train += tn
            FN_train += fn
        
            if np.mod(index, 1) ==0:
                print('epoch {}, {}/{}, loss is {}, name is {}'.format(epo, index, len(dataloader), iter_loss, name))
                #vis.close()
                vis.images(output_np[:, None, :, :], win = 'predict', opts=dict(title='pred')) 
                vis.images(y_np[:, None, :, :], win = 'groundtruth', opts=dict(title='label'))
                #vis.images(x555, opts = dict(title='feature'))
                #vis.images(input[:,:,:,:], opts = dict(title='original'))
            #plt.subplot(1, 2, 1) 
            #plt.imshow(np.squeeze(y_np[0, :, :]), 'gray')
            #plt.subplot(1, 2, 2) 
            #plt.imshow(np.squeeze(output_np[0, :, :]), 'gray')
            #plt.pause(0.5)
            
        if TP_train + FP_train + TN_train +FN_train == (output_np.shape[2] * output_np.shape[1] *53):
            accuray = (TP_train + TN_train)/(TP_train+TN_train+FP_train+FN_train)
            precision = (TP_train)/(TP_train+FP_train)
            recall = TP_train/(TP_train+FN_train)
        else:
            print('wrong evaluation training')
            print(output_np.shape, output_np.shape[1], len(dataloader), dataloader.__len__() )
            
        F1_training = 2*((precision*recall)/(precision+recall))
            
        print('epoch loss = %f'%(epo_loss/len(dataloader)), 'f1 = %f'%(F1_training))
        epoloss = epo_loss/len(dataloader)
        epoloss_np = epoloss.cpu().data.numpy().copy()
        
        with torch.no_grad():
            fcn_model.eval()
            F1 = check_accuracy(fcn_model)
       
        fcn_model.train()
       
        
#         F1 = check_accuracy(fcn_model)
        vis.line(X = np.array([epo]), Y = np.array([F1]), win = 'f1-score' , name='testing', update='append',opts=dict(title='f1-score'),)
        vis.line(X = np.array([epo]), Y = np.array([F1_training]), win = 'f1-score' , name='training', update='append',opts=dict(title='f1-score'),)
        vis.line(X = np.array([epo]), Y = np.array([epoloss_np]), win = 'loss' , name='line1', update='append',opts=dict(title='loss'),)
        
        if np.mod(saving_index, 1)==0:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pt'.format(epo))
