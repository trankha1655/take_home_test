import torch.utils.data as data 
import torch
import os
from PIL import Image
import numpy as np
import cv2

class MyDataset(data.Dataset):
    def __init__(self, data_dir, annotation_file, transform=None, aug_transform=None, input_size=512, classes=[]):
        pass 

        self.data_list = [x.strip().split('\t') for x in open(annotation_file)]
        self.classes = classes
        self.input_size = input_size
        self.transform = transform
        self.data_dir = data_dir
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.data_list)


    def resize_padding(self, image):
        #Resize padding  
        H, W = image.shape[:2]
        max_side = max(H, W)
        img_pad = np.zeros((max_side, max_side, 3))
        img_pad[(max_side-H)//2: (max_side-H)//2 + H, (max_side-W)//2: (max_side-W)//2 + W, :] = image 
        img_pad = img_pad.astype(np.uint8)
        img_resized = cv2.resize(img_pad, (self.input_size, self.input_size))

        return img_resized
    
    def get_image(self, img_path):

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        img_resized = self.resize_padding(img)

        return img_resized


        

    def __getitem__(self, index):
        img_path, cls_name = self.data_list[index]
        img_path = os.path.join(self.data_dir, img_path)
        if not os.path.exists(img_path):
            print("Found nothing from", img_path)
            return self.__getitem__(min(index + 1, len(self.classes)))
        

        image = self.get_image(img_path)
        image = image if not self.aug_transform else self.aug_transform(image=image)['image']
        tensor = image if not self.transform else self.transform(image)
        label_cls = self.classes[cls_name]

        #one hot encoding
        label_onehot = np.zeros((len(self.classes))) 
        label_onehot[label_cls] = 1

        label_onehot = torch.from_numpy(label_onehot).float()


        return tensor, label_cls, label_onehot, img_path
        


        





    