import os 
import numpy as np 
import glob 
import random

dir = "/home/khatpn/KT/take_home_test/datasets/"
train_dir = os.path.join(dir, "data_train")
test_dir = os.path.join(dir, "data_test")

train_list = []
val_list = []
for cls in os.listdir(train_dir):
    path = os.path.join(train_dir, cls, "*")
    # print(path)
    files = glob.glob(path)
    files = [f"{x.replace(dir, '')}\t{cls}" for x in files]
    random.shuffle(files) 
    n = len(files)
    partition = int(0.9*n)

    train_list += files[:partition]
    val_list += files[partition:]

files = glob.glob(test_dir + "/*/*")
test_list = [f"{x.replace(dir, '')}\t{x.split('/')[-2]}" for x in files]

print("Length train:", len(train_list))
print("Length val:", len(val_list))
print("Length test:", len(test_list))

with open(dir + "test.txt", 'w') as f:
    f.write("\n".join(test_list))

with open(dir + "train.txt", 'w') as f:
    f.write("\n".join(train_list))

with open(dir + "val.txt", 'w') as f:
    f.write("\n".join(val_list))


    
