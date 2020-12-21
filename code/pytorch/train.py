import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from dataset import TrainDataset
from model import Resnet

### VIZ ###
writer = SummaryWriter(filename_suffix="512")
ppepoch = 10 #points per epoch

### HYPERPARAMETERS ###
batch_size = 3
lr = 1e-4
weight_decay = 1e-6
num_epoch = 100
save_path = "pytorch/model_512.pth"

### MODEL ###
model = Resnet().cuda()

### DATASET ###
dataset = TrainDataset()
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

datasetVal = TrainDataset(is_val=True)
dataloaderVal = DataLoader(
    datasetVal,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

### OPTIM ###
optim = torch.optim.Adam(
    model.parameters(),
    lr = lr,
    weight_decay=weight_decay
)

### LOAD ###
if os.path.isfile(save_path):
    m = torch.load(save_path)
    model.load_state_dict(m["model"])
    optim.load_state_dict(m["optim"])
    del m

### LOSS ###
loss = torch.nn.L1Loss()

benchmark_img = dataset.transform("DanbooRegion2020/train/0.image.png").unsqueeze(0).cuda()
benchmark_skel = dataset.transform("DanbooRegion2020/train/0.skeleton.png").unsqueeze(0).expand(1,3,-1,-1).cuda()
len_dataloader = len(dataloader)
len_dataloader_val = len(dataloaderVal)
for e in range(206, num_epoch+206):
    pbar = tqdm(enumerate(dataloader), total=len_dataloader)
    with torch.no_grad():
        writer.add_image(
            'image',
            torchvision.utils.make_grid(
                torch.cat(
                    (benchmark_img,
                    model(benchmark_img).expand(1,3,-1,-1),
                    benchmark_skel),
                    dim=-1
                )
            ),
            e
        )
    for idx, (img, skel) in pbar:
        y = model(img.cuda())
        l = loss(y, skel.cuda())
        l.backward()
        optim.step()
        optim.zero_grad()
        if not idx % (len_dataloader//ppepoch):
            writer.add_scalar('Loss/train', l.item(), e*len_dataloader + idx)
        
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict()
        },
        save_path[:-4]+'_'+str(e)+'.pth'
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict()
        },
        save_path
    )

    pbar = tqdm(enumerate(dataloaderVal), total=len_dataloader_val)
    l = 0
    with torch.no_grad():
        for idx, (img, skel) in pbar:
            y = model(img.cuda())
            l += loss(y, skel.cuda()).item()
            
        writer.add_scalar('Loss/val', l/len_dataloader_val, e*len_dataloader_val)