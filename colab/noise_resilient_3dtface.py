# -*- coding: utf-8 -*-
"""noise_resilient_3dtface

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17yobL9qT4BtAZR7vI-HriRMzXzcEct9d
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/eeskimez/noise_resilient_3dtface
# %cd noise_resilient_3dtface
!ls

"""from google.colab.patches import cv2_imshow
cv2_imshow(img)
pred = depth_to_grayscale(depth)
cv2_imshow(pred)
cv2.imwrite("pred.png",pred)
"""


import torch;
import torch.nn as nn
import torch.nn.functional as F

class SPCH2FLM(nn.Module):
    def __init__(self, numFilters=64, filterWidth=21):
        super(SPCH2FLM, self).__init__()
        self.numFilters = numFilters
        self.filterWidth = filterWidth
        self.conv1 = nn.Conv1d(1, self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)
        self.conv2 = nn.Conv1d(self.numFilters, 2*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)     
        self.conv3 = nn.Conv1d(2*self.numFilters, 4*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)
        self.conv4 = nn.Conv1d(4*self.numFilters, 8*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)
        self.fc1 = nn.Linear(62464, 6) 

    def forward(self, x):
        h = F.dropout(F.leaky_relu(self.conv1(x), 0.3), 0.2)
        h = F.dropout(F.leaky_relu(self.conv2(h), 0.3), 0.2)
        h = F.dropout(F.leaky_relu(self.conv3(h), 0.3), 0.2)
        h = F.dropout(F.leaky_relu(self.conv4(h), 0.3), 0.2)
        features = h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.fc1(h), 0.3)
        return h, features


def convert_to_onnx(net, output_name):
    input = torch.randn(1, 1, 2240)
    input_names = ['data']
    output_names = ['output']
    net.eval()
    torch.onnx.export(net, input, output_name, verbose=True, input_names=input_names, output_names=output_names)

model = SPCH2FLM()
model.load_state_dict(torch.load("/content/noise_resilient_3dtface/pre_trained/1D_CNN.pt"))

convert_to_onnx(model, "model.onnx")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/noise_resilient_3dtface/code
!rm  -f /content/noise_resilient_3dtface/code/__pycache__/*
!python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN.pt -o ../results/1D_CNN/
# %cd /content/noise_resilient_3dtface

!rm  -f /content/noise_resilient_3dtface/code/__pycache__/*

import cv2
net = cv2.dnn.readNet("model.onnx")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/noise_resilient_3dtface
!cp "/content/drive/My Drive/cv2_cuda/cv2.cpython-36m-x86_64-linux-gnu.so" .

