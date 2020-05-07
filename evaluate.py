#! /usr/bin/env python3

import sys

import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data_helper import LabeledDataset
from helper import compute_ats_bounding_boxes, compute_ts_road_map

from model_loader import get_transform_task1, get_transform_task2, ModelLoader


class Logger:
    # https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('kobe_log.log', 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    


sys.stdout = Logger()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--filename', type=str, default='kobe_model_w_pretrain2_9_epochs.pt')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--prob_thresh', type=float, default=0.1)
parser.add_argument('--conf_thresh', type=float, default=0.1)
parser.add_argument('--nms_thresh', type=float, default=0.4)
parser.add_argument('--batch_norm', action = 'store_true')
opt = parser.parse_args()

print(f'Args: {opt}')

image_folder = opt.data_dir
annotation_csv = f'{opt.data_dir}/annotation.csv'

labeled_scene_index = np.arange(120, 134)

# For bounding boxes task
labeled_trainset_task1 = LabeledDataset(
    image_folder=image_folder,
    annotation_file=annotation_csv,
    scene_index=labeled_scene_index,
    transform=get_transform_task1(),
    extra_info=False
    )
dataloader_task1 = torch.utils.data.DataLoader(
    labeled_trainset_task1,
    batch_size=1,
    shuffle=False,
    num_workers=0
    )
# For road map task
labeled_trainset_task2 = LabeledDataset(
    image_folder=image_folder,
    annotation_file=annotation_csv,
    scene_index=labeled_scene_index,
    transform=get_transform_task2(),
    extra_info=False
    )
dataloader_task2 = torch.utils.data.DataLoader(
    labeled_trainset_task2,
    batch_size=1,
    shuffle=False,
    num_workers=0
    )

model_loader = ModelLoader(model_file=opt.filename,
                           prob_thresh=opt.prob_thresh,
                           conf_thresh=opt.conf_thresh,
                           nms_thresh=opt.nms_thresh,
                           batch_norm=opt.batch_norm,
                           )

model_loader.model.eval()

print(model_loader)

total = 0
total_ats_bounding_boxes = 0
total_ts_road_map = 0

with torch.no_grad():
    for i, data in enumerate(dataloader_task1):
        total += 1
        sample, target, road_image = data
        sample = sample.cuda()

        predicted_bounding_boxes = model_loader.get_bounding_boxes(sample)[0].cpu()
        ats_bounding_boxes, iou_max = compute_ats_bounding_boxes(predicted_bounding_boxes, target['bounding_box'][0])
        total_ats_bounding_boxes += ats_bounding_boxes

        if opt.verbose:
            print(f'{i} - Bounding Box Score: {ats_bounding_boxes:.4}')
            print(f'{i} - IOU_max: {iou_max}')

    print('Finished testing bounding box')

    for i, data in enumerate(dataloader_task2):
        sample, target, road_image = data
        sample = sample.cuda()

        predicted_road_map = model_loader.get_binary_road_map(sample).cpu()
        ts_road_map = compute_ts_road_map(predicted_road_map, road_image)
        total_ts_road_map += ts_road_map

        if opt.verbose:
            print(f'{i} - Road Map Score: {ts_road_map:.4}')

        print()

    print('Finished testing road map')

print(f'{model_loader.team_name} - {model_loader.round_number} - Bounding Box Score: {total_ats_bounding_boxes / total:.4} - Road Map Score: {total_ts_road_map / total:.4}')
print('Max bounding box score: 1.0, Max roadmap score: 1.0')
