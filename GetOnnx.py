import os
import torch
from models.MIMOUNet import build_net

model = build_net('MIMO-UNetPlus')
state_dict = torch.load('MIMO-UNetPlus.pkl')
model.load_state_dict(state_dict['model'])
torch.onnx.export(model, torch.randn(1,3,720,1280), './MIMO_Tensorrt/weights/MIMO.onnx', verbose=False)

