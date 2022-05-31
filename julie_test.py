import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, glob
import cv2
import pdb
from onehot import onehot
import torch
from BagData import input_h, input_w 
from torch.autograd import Variable
import numpy as np
from FCN import FCNs
from FCN import VGGNet 

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class BagDataset(Dataset):

    def __init__(self, root_path, transform=None):
        self.rgblist = glob.glob(os.path.join(root_path, '*.jpg'))
        self.transform = transform

    def __len__(self):
       return len(self.rgblist)

    def __getitem__(self, idx):
        imgA = cv2.imread(self.rgblist[idx])
        imgA = imgA[300:1000,500:1500]
        basename = os.path.basename(self.rgblist[idx])
        print(self.rgblist[idx], basename)
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgA = cv2.resize(imgA, (input_w, input_h))

        if self.transform:
            imgA_t = self.transform(imgA)    
        #item = {'A':imgA, 'B':imgB}
        item = {'A':imgA_t, 'B': imgA, 'C': basename}
        return item

root = 'quads'
des = 'quads_mask'
if not os.path.exists(des):
    os.mkdir(des)
    
bag_test = BagDataset(root, transform)
dataloader_test = DataLoader(bag_test, batch_size=1, shuffle=True, num_workers=0)
model = torch.load('checkpoints/fcn_model_24.pt')
model = model.cuda()
print(model)
model.eval()
for item in dataloader_test:
    input = item['A']
    name = item['C']
    name = np.array(name)
    input = Variable(input)
    input = input.cuda()
    
    output = model(input)

    output_np = output.cpu().detach().numpy()
    output_np = np.argmin(output_np, axis=1) 
    output_np = output_np.reshape((input_h,input_w))
    output_test = np.uint8(output_np*255)
#     output_test = cv2.resize(output_test, (1936, 1216))
#     name = np.array(name)
#     basename = os.path.basename(name)
#     print(basename)
    cv2.imwrite(os.path.join(des, name[0]), output_test)
    del input, output


# if __name__ =='__main__':
#     for batch in dataloader_test:
#         break