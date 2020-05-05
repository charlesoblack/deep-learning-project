"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
from src import *

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    # return torchvision.transforms.Compose([
    # 
    # 
    # ])
    pass

def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = ''
    team_member = []
    contact_email = '@nyu.edu'

    def __init__(model_file):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        
        self.model = KobeModel(num_classes = 10, encoder_features = 6, rm_dim = 800)
        
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(self.device)
        

    def get_bounding_boxes(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        samples.to(self.device)
        boxes, _ = self.model.get_bounding_boxes(samples)
        
        # converted to cuda already inside
        return boxes

    def get_binary_road_map(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        samples.to(self.device)
        road_map, _ = self.model.get_road_map(samples)
        
        return road_map.cuda()
