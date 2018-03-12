####DEFINE CUSTOM LOSS FUNCTIONS HERE



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

from PIL import Image
from argparse import ArgumentParser

from utils import *
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)

class ConsistencyLoss(nn.Module):

    def __init__(self, weight=None):
        super(ConsistencyLoss, self).__init__()

        # self.loss = nn.NLLLoss2d(weight)
    def scale_dims(self, x, orig_dim, scaledim):
        return int(x*(scaledim/orig_dim))

    def forward(self, outputs, target, bbox, orig_dims):
        loss = Variable(torch.FloatTensor([0.0]).cuda())
	num_bboxes = 0
        for i in range(0,outputs.shape[0]):    
            image_height = orig_dims[i][0]
            image_width = orig_dims[i][1] 
            for boxes in bbox[i].values():
                for box in boxes:
                    ymin = self.scale_dims(box[0], image_height, 800.0)
                    ymax = self.scale_dims(box[1], image_height, 800.0)
                    xmin = self.scale_dims(box[2], image_width, 400.0)
                    xmax = self.scale_dims(box[3], image_width, 400.0)
                    outputs_matrix = outputs[i,:,ymin:ymax,xmin:xmax]
		    outputs_matrix = outputs_matrix.contiguous().view(outputs.shape[1],-1)
		    avg = torch.mean(outputs_matrix, dim=1)
                    diff = outputs_matrix.transpose(1,0) - avg
                    pixloss = torch.pow(diff, 2)
                    pixloss = torch.sum(pixloss)/((ymax- ymin) * (xmax - xmin) * outputs.shape[1])
		    loss += torch.sum(pixloss)
                    num_bboxes+=1
        return loss/num_bboxes
    
