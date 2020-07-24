"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
from src import KobeModel

# Put your transform function here, we will use it for our dataloader
def get_transform():
    # return torchvision.transforms.Compose([
    #
    #
    # ])
    pass

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1():
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2():
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'Los Tres Latinos'
    team_number = 13
    round_number = 3
    team_member = ['Nabeel Sarwar', 'Esteban Navarro Garaiz', 'Guido Petri']
    contact_email = 'gp1655@nyu.edu'

    def __init__(self, model_file='combined_model.pt', prob_thresh=0.1, conf_thresh=0.1, nms_thresh=0.4, batch_norm=False, shared_decoder=False):

        # You should
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...

        self.model = KobeModel(num_classes=10,
                               encoder_features=6,
                               rm_dim=800,
                               prob_thresh=prob_thresh,
                               conf_thresh=conf_thresh,
                               nms_thresh=nms_thresh,
                               batch_norm=batch_norm,
                               shared_decoder=shared_decoder
                               )

        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)


    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        samples.to(self.device)
        boxes, _ = self.model.get_bounding_boxes(samples)

        return boxes

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]

        samples.to(self.device)
        road_map, _ = self.model.get_road_map(samples)

        # binarize for a better score
        road_map = road_map > 0.5

        return road_map
