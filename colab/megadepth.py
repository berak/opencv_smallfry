# -*- coding: utf-8 -*-
"""MegaDepth.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FPZnCH7SqW0DkNk6MlNAjGtceaOXmQ0s
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/zhengqili/MegaDepth
# %cd MegaDepth

!wget http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth

!mkdir checkpoints
!mkdir checkpoints/test_local
!cp best_generalization_net_G.pth checkpoints/test_local

!ls -l checkpoints/test_local

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/MegaDepth
!python demo.py

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/MegaDepth
import cv2
net = cv2.dnn.readNet("megadepth.onnx")

!cp "/content/drive/My Drive/cv2_cuda/cv2.cpython-37m-x86_64-linux-gnu.so" .

import torch
from collections import OrderedDict

def key_transformation(old_key):
    if old_key.find("module")>=0:
        return old_key[7:]
    return old_key

def rename_state_dict_keys(source, key_transformation, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)

rename_state_dict_keys("checkpoints/test_local/best_generalization_net_G.pth", key_transformation)


# demo.py
"""
import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize


img_path = 'demo.jpg'

model = create_model(opt)

input_height = 384
input_width  = 512


def test_simple(model):
    total_loss =0
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()

    img = np.float32(io.imread(img_path))/255.0
    img = resize(img, (input_height, input_width), order = 1)
    input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
    input_img = input_img.unsqueeze(0)

    input_images = Variable(input_img.cpu() )
    pred_log_depth = model.netG.forward(input_images)
    pred_log_depth = torch.squeeze(pred_log_depth)

    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

    io.imsave('demo.png', pred_inv_depth)
    # print(pred_inv_depth.shape)

def save_onnx(model):
    model.eval()
    with torch.no_grad():
      dummy_input = torch.randn(1,3,input_height, input_width, device='cpu')
      out = model.forward(dummy_input)
      print(out.shape)
      torch.onnx.export(model, dummy_input, "megadepth.onnx", opset_version=10, verbose=True, input_names=["input"], output_names=["output"])

print(model.netG)
save_onnx(model.netG)
test_simple(model)
print("We are done")

"""



# models/HG_model.py
"""
import numpy as np
import torch
import os
from torch.autograd import Variable
from .base_model import BaseModel
import sys
import pytorch_DIW_scratch

class HGModel(BaseModel):
    def name(self):
        return 'HGModel'

    def __init__(self, opt):
        BaseModel.initialize(self, opt)

        print("===========================================LOADING Hourglass NETWORK====================================================")
        model = pytorch_DIW_scratch.pytorch_DIW_scratch
        # model= torch.nn.parallel.DataParallel(model, device_ids = [0,1])
        model_parameters = self.load_network(model, 'G', 'best_generalization')
        model.load_state_dict(model_parameters)
        self.netG = model.cpu()


    def batch_classify(self, z_A_arr, z_B_arr, ground_truth ):
        threashold = 1.1
        depth_ratio = torch.div(z_A_arr, z_B_arr)

        depth_ratio = depth_ratio.cpu()

        estimated_labels = torch.zeros(depth_ratio.size(0))

        estimated_labels[depth_ratio > (threashold)] = 1
        estimated_labels[depth_ratio < (1/threashold)] = -1

        diff = estimated_labels - ground_truth
        diff[diff != 0] = 1

        # error
        inequal_error_count = diff[ground_truth != 0]
        inequal_error_count =  torch.sum(inequal_error_count)

        error_count = torch.sum(diff) #diff[diff !=0]
        # error_count = error_count.size(0)

        equal_error_count = error_count - inequal_error_count


        # total
        total_count = depth_ratio.size(0)
        ground_truth[ground_truth !=0 ] = 1

        inequal_count_total = torch.sum(ground_truth)
        equal_total_count = total_count - inequal_count_total


        error_list = [equal_error_count, inequal_error_count, error_count]
        count_list = [equal_total_count, inequal_count_total, total_count]

        return error_list, count_list


    def computeSDR(self, prediction_d, targets):
        #  for each image
        total_error = [0,0,0]
        total_samples = [0,0,0]

        for i in range(0, prediction_d.size(0)):

            if targets['has_SfM_feature'][i] == False:
                continue

            x_A_arr = targets["sdr_xA"][i].squeeze(0)
            x_B_arr = targets["sdr_xB"][i].squeeze(0)
            y_A_arr = targets["sdr_yA"][i].squeeze(0)
            y_B_arr = targets["sdr_yB"][i].squeeze(0)

            predict_depth = torch.exp(prediction_d[i,:,:])
            predict_depth = predict_depth.squeeze(0)
            ground_truth = targets["sdr_gt"][i]

            # print(x_A_arr.size())
            # print(y_A_arr.size())

            z_A_arr = torch.gather( torch.index_select(predict_depth, 1 ,x_A_arr.cuda()) , 0, y_A_arr.view(1, -1).cuda())# predict_depth:index(2, x_A_arr):gather(1, y_A_arr:view(1, -1))
            z_B_arr = torch.gather( torch.index_select(predict_depth, 1 ,x_B_arr.cuda()) , 0, y_B_arr.view(1, -1).cuda())

            z_A_arr = z_A_arr.squeeze(0)
            z_B_arr = z_B_arr.squeeze(0)

            error_list, count_list  = self.batch_classify(z_A_arr, z_B_arr,ground_truth)

            for j in range(0,3):
                total_error[j] += error_list[j]
                total_samples[j] += count_list[j]

        return  total_error, total_samples


    def evaluate_SDR(self, input_, targets):
        input_images = Variable(input_.cpu() )
        prediction_d = self.netG.forward(input_images)

        total_error, total_samples = self.computeSDR(prediction_d.data, targets)

        return total_error, total_samples

    def rmse_Loss(self, log_prediction_d, mask, log_gt):
        N = torch.sum(mask)
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)
        s1 = torch.sum( torch.pow(log_d_diff,2) )/N

        s2 = torch.pow(torch.sum(log_d_diff),2)/(N*N)
        data_loss = s1 - s2

        data_loss = torch.sqrt(data_loss)

        return data_loss

    def evaluate_RMSE(self, input_images, prediction_d, targets):
        count = 0
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        mask_0 = Variable(targets['mask_0'].cuda(), requires_grad = False)
        d_gt_0 = torch.log(Variable(targets['gt_0'].cuda(), requires_grad = False))

        for i in range(0, mask_0.size(0)):

            total_loss +=  self.rmse_Loss(prediction_d[i,:,:], mask_0[i,:,:], d_gt_0[i,:,:])
            count += 1

        return total_loss.data[0], count


    def evaluate_sc_inv(self, input_, targets):
        input_images = Variable(input_.cuda() )
        prediction_d = self.netG.forward(input_images)
        rmse_loss , count= self.evaluate_RMSE(input_images, prediction_d, targets)

        return rmse_loss, count


    def switch_to_train(self):
        self.netG.train()

    def switch_to_eval(self):
        self.netG.eval()

"""
