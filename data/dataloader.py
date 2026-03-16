

import torch
import nni
import os
import numpy as np
import albumentations as A

from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import MyDataset
from torch.utils.data.distributed import DistributedSampler


class MyDataModule:
    def __init__(self, config):
        self.config = config

        train_data_path = config['train_data_path']
        val_data_path = config['val_data_path']
        test_data_path = config['test_data_path']
        data_dir = config['data_dir']
        classes_name = config['classes_name']

        input_size = config["input_size"]
        num_workers = config['num_workers']
        batch_size = config["batch_size"]

        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
            ]
        )    
        # Augmentation
        self.aug_transform = A.Compose([
            A.OneOf([
                A.RandomSizedCrop(
                    min_max_height=(int(input_size*0.95), input_size), 
                    height=input_size, 
                    width=input_size, 
                    p=0.2
                ),
            ], p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.RandomGamma(p=0.2),
            A.RandomBrightnessContrast(0.2),
            A.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.1, always_apply=False, p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
            ], p=0.2),
            A.OneOf([
                A.CoarseDropout(min_holes=1, max_holes=2, max_height=int(input_size*0.3), max_width=int(input_size*0.3), min_height=int(input_size*0.1), min_width=int(input_size*0.1), p=0.3),
                A.CoarseDropout(min_holes=1, max_holes=2, max_height=int(input_size*0.3), max_width=int(input_size*0.3), min_height=int(input_size*0.1), min_width=int(input_size*0.1), fill_value=255, p=0.1),
            ], p=0.5),
        ])


        train_dataset = MyDataset(data_dir, train_data_path, self.transform, self.aug_transform, input_size=input_size, classes=classes_name)
        val_dataset = MyDataset(data_dir, val_data_path, self.transform, input_size=input_size, classes=classes_name)
        test_dataset = MyDataset(data_dir, test_data_path, self.transform, input_size=input_size, classes=classes_name)



        self.train_loader = DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 shuffle=True,
                                 drop_last=True, 
                                 sampler=None
                                 )
        
        self.val_loader = DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 drop_last=True, 
                                 sampler=None
                                 )
        
        self.test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 drop_last=False, 
                                 sampler=None
                                 )

    @property
    def train(self):
        return self.train_loader

    @property
    def val(self):
        return self.val_loader
    
    
    @property
    def test(self):
        return self.test_loader
    


if __name__ == "__main__":
    pass 

    config = {
        "train_data_path": "./datasets/train.txt",
        "val_data_path": "./datasets/val.txt",
        "test_data_path": "./datasets/test.txt",
        "data_dir": "./datasets",

        "input_size": 512,
        "num_workers": 2,
        "batch_size": 4,
    }

    classes = ["em_bé_chơi_verified", "ngày_tết_verified", "other", "thiennhien", "trekking_verified", "tụ_họp_verified"]
    classes = {x: i for i, x in enumerate(classes)}

    # print(classes, config)
    config["classes_name"] = classes

    datamodule = MyDataModule(config)
    train_loader = datamodule.train 
    train_datasets = train_loader.dataset

    import cv2
    import tqdm

    vis_grid = (10, 10)  # Define the grid size for visualization
    count = 0 
    shape = (224, 224)  # Define the shape of the images 
    vis_image = np.zeros((shape[0] * vis_grid[0], shape[1] * vis_grid[1], 3), dtype=np.uint8)
    
    for iter, batch in enumerate(tqdm.tqdm(train_datasets)):

        tensor, label, path = batch
        
        # map = cv2.resize(map, (192, 256))

        image = tensor.cpu().numpy() 
        image = image.transpose(1, 2, 0)  # Convert to HWC format  
        image = (image * 0.5) + 0.5 
        image *= 255
        image = cv2.resize(image, (shape))

        
 
        row = count // vis_grid[1]
        col = count % vis_grid[1]
        y_start = row * shape[0]
        y_end = y_start + shape[0]
        x_start = col * shape[1]
        x_end = x_start + shape[1]
        
        vis_image[y_start:y_end, x_start:x_end] = image.astype(np.uint8)  # Place the blended image in the grid
        count += 1
        if count >= vis_grid[0] * vis_grid[1]:
            cv2.imwrite(f'assets/train_data_aug_{iter // (vis_grid[0] * vis_grid[1])}.jpg', vis_image[:,:,::-1])
            count = 0
