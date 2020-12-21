import cv2
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TrainDataset(Dataset):
    def __init__(self, max_num_pic=None, load_in_mem=False, is_val=False):
        self.load_in_mem = load_in_mem
        if is_val:
            self.folder = "DanbooRegion2020/val"
        else:
            self.folder = "DanbooRegion2020/train"
        self.files = os.listdir(self.folder)[:max_num_pic]
        self.len = len(self.files)//3

        if load_in_mem:
            print("loading images...")
            self.img = [
                self.transform(self.folder+"/"+str(i)+".image.png")
                for i in tqdm(range(self.len))
            ]

            print("loading skeletons...")
            self.skel = [
                self.transform(self.folder+"/"+str(i)+".skeleton.png").mean(dim=0, keepdim=True)
                for i in tqdm(range(self.len))
            ]
    
    def transform(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512,512))
        img = img/255
        img = torch.from_numpy(img.transpose(2,0,1))
        img = img.type(torch.float)
        return img
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.load_in_mem:
            return self.img[idx], self.skel[idx]
        else:
            return self.transform(self.folder+"/"+str(idx)+".image.png"), self.transform(self.folder+"/"+str(idx)+".skeleton.png").mean(dim=0, keepdim=True)