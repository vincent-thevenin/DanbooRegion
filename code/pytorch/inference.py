import matplotlib.pyplot as plt
import os
import torch

from dataset import TrainDataset
from model import Resnet

path = "pytorch/model_l1_128.pth"

### MODEL ###
model = Resnet().cuda()
model = model.eval()

### DATASET ###
dataset = TrainDataset(max_num_pic=3)

### LOAD ###
if os.path.isfile(path):
    m = torch.load(path)
    model.load_state_dict(m["model"])
    del m

benchmark_img = dataset.transform("DanbooRegion2020/train/0.image.png").unsqueeze(0).cuda()
benchmark_skel = dataset.transform("DanbooRegion2020/train/0.skeleton.png").unsqueeze(0).expand(1,3,-1,-1).cuda()
y = model(benchmark_img)
plt.imsave("pytorch/test.png", -y[0,0].detach().cpu().numpy()+1, cmap='Greys')
