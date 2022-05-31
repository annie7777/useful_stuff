import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import pdb
from onehot import onehot
import torch
import random
import numpy as np

input_h = 960
input_w = 960

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
class BagDataset(Dataset):

    def __init__(self, transform=None):
       self.transform = transform 
    def __len__(self):
       return len(os.listdir('training_img'))

    def __getitem__(self, idx):
        img_name = os.listdir('training_img')[idx]
        #print(img_name)
        imgA = cv2.imread('training_img/'+img_name)
        
        
        #crop
        imgA = imgA[300:1000,500:1500, :]
        imgA = cv2.resize(imgA, (input_w, input_h))
        
        imgB = cv2.imread('training_label/'+img_name, 0)  
        imgB = imgB[300:1000,500:1500]
        
        imgB = cv2.resize(imgB, (input_w, input_h))
        imgB[imgB>0] = 255
        
#         ret, thresh = cv2.threshold(imgB, 10, 255, cv2.THRESH_BINARY)
#         #print(np.unique(imgB))
#         cv2.imwrite('test_results/'+('training'+img_name),imgB) 

        #rand = random.random()
        #print(rand)
        #print('1111.......')
#         if rand > 0.5:
#             imgA = cv2.flip(imgA, 0 )
#             imgB = cv2.flip(imgB, 0)
#         #print(imgA)

        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.swapaxes(0, 2).swapaxes(1, 2)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        item = {'A':imgA, 'B':imgB, 'C':img_name}

            
        return item

bag = BagDataset(transform)
dataloader = DataLoader(bag, batch_size=2, shuffle=True, num_workers=0)
if __name__ =='__main__':
    for batch in dataloader:
        break






