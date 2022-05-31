import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import pdb
from onehot import onehot
import torch
from BagData import input_h, input_w

#input_h = 1280
#input_w = 1920

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

class BagDataset(Dataset):

    def __init__(self, transform=None):
       self.transform = transform 
    def __len__(self):
       return len(os.listdir('testing_img'))

    def __getitem__(self, idx):
        
        img_name = os.listdir('testing_img')[idx]
      
        print(img_name)
        imgA = cv2.imread('testing_img/'+img_name)
        imgA = imgA[300:1000,500:1500, :]
        imgA = cv2.resize(imgA, (input_w, input_h))
        #imgB = cv2.imread('gt_test/'+img_name, 0)
        #imgB = cv2.resize(imgB, (1920, 1280))
        #imgB = imgB/255
        #imgB = imgB.astype('uint8')
        #imgB = onehot(imgB, 2)
        #imgB = imgB.swapaxes(0, 2).swapaxes(1, 2)
        #imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA_t = self.transform(imgA)    
        #item = {'A':imgA, 'B':imgB}
        item = {'A':imgA_t, 'B': imgA, 'C': img_name}
        return item

bag_test = BagDataset(transform)
dataloader_test = DataLoader(bag_test, batch_size=1, shuffle=True, num_workers=0)
if __name__ =='__main__':
    for batch in dataloader_test:
        break