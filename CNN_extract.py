import os
import operator
import torch
#import tensorflow as tf
#from tensorflow import metrics
from net import *
from utils import keep_image_size_open
from data import *
from torchvision.utils import save_image

device=torch.device('cuda')
net=UNet().to(device)

weights='params/unet.pth'
net.load_state_dict(torch.load(weights))
path='test_image/'  #image file path
savepath='test_image/'  #segmented file save path

filenames = os.listdir(path)
for name in filenames:
    file = path + "\\" + name
    _input = file
    img = keep_image_size_open(_input)
    img_data = transform(img).to(device)
    save_image(img_data, savepath+"\\"+'reshape'+name)
    img_data = torch.unsqueeze(img_data, dim=0)
    out = net(img_data)
    save_image(out, savepath+'\\'+'result'+name)



