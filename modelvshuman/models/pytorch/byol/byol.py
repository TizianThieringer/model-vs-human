import torch
from torchvision import models
import os

print(os.getcwd())

state_dict = torch.load('improved-net.pt')


def get_model():
    
    print(os.getcwd())
    model = models.resnet50()
    model.load_state_dict(state_dict)
        
    return model