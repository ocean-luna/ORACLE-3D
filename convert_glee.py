# Copyright (c) 2024 ByteDance. All Rights Reserved
"""
Helper script to convert models trained with the main version of DETR to be used with the Detectron2 version.
"""
import json
import argparse

import numpy as np
import torch

# lang = torch.load('converted_seem_focalt.pt', map_location="cpu")

model_to_convert = torch.load('model_path/GLEE_Pro_scaleup.pth', map_location="cpu")
model_converted = {}
remove_list = []
for k in model_to_convert.keys():

    if 'glee.pixel_decoder' in k or 'glee.predictor' in k:
        newk = "sem_seg_head"+k[4:]
        model_converted[newk] = model_to_convert[k].detach()
    elif 'glee' in k:
        newk = k[5:]
        model_converted[newk] = model_to_convert[k].detach()

model_to_save = {"model": model_converted}
torch.save(model_to_save, 'model_path/GLEE_Pro_scaleup_convert.pth')


