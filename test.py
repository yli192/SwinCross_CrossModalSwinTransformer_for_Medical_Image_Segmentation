# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
from monai.inferers import sliding_window_inference
import nibabel as nib
from utils.data_utils import get_loader
from trainer import dice
import argparse
from SwinTransModels import CONFIGS as CONFIGS_sw_seg
from SwinTransModels import *
import warnings
import cv2
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(99999)
print(sys.getrecursionlimit())

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='/home/local/PARTNERS/yl715/segmentationScratch/research-contributions/UNETR/BTCV/runs/SWINnetr_DualModalityCrossAttnCMFF_hector_nnUnetSplit_2channel_C48_SumOuts_6Outs_v06_2422/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/home/local/PARTNERS/yl715/Task107_hecktor2021/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_sameSplit_nnUNet_2channel.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='model.pt', type=str, help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=36, type=int, help='feature size dimention')
parser.add_argument('--infer_overlap', default=0.4, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=2, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=2, type=int, help='number of output channels')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--lower', default=0.0, type=float, help='a_min in ScaleIntensityRangePercentilesd')
parser.add_argument('--upper', default=100.0, type=float, help='a_max in ScaleIntensityRangePercentilesd')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRangePercentilesd')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRangePercentilesd')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')

def count_objects(image):
    width, height, depth = image.shape[0], image.shape[1], image.shape[2]
    objects_members = [(x, y, z) for x in range(width) for y in range(height) for z in range(depth) if image[y][x][z] == 1]
    objects_count = 0
    while objects_members != []:
       remove_object(objects_members, objects_members.pop(0))
       objects_count += 1
    return objects_count

def remove_object(objects_members, start_point):
    x, y ,z = start_point
    connex = [(x, y + 1,z+1), (x - 1, y, z), (x, y - 1,z),(x, y + 1,z-1),
              (x + 1, y, z), (x, y + 1,z)]
    for point in connex:
        try:
            objects_members.remove(point)
            remove_object(objects_members, point)
        except ValueError:
            pass

def main():
    args = parser.parse_args()
    args.test_mode = True
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == 'torchscript':
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == 'ckpt':
        # config_sw = CONFIGS_sw_seg['SwinUNETR_CMFF-hecktor-v01']
        # model = SwinUNETR_fusion(config_sw)
        config_sw = CONFIGS_sw_seg['SwinUNETR_CMFF-hecktor-v06']
        model = SwinUNETR_CrossModalityFusion_OutSum_6stageOuts(config_sw)
        model_dict = torch.load(pretrained_pth)['state_dict']
        model.load_state_dict(model_dict)

    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        if not os.path.exists(pretrained_dir + 'test'):
            os.makedirs(pretrained_dir + 'test')
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            #val_inputs = val_inputs[:,1,:,:,:]
            #val_inputs = torch.unsqueeze(val_inputs,1)
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            img_prefix = img_name.split('.')[0]

            #print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96),
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]

            tumor_volume = np.sum(np.sum(np.sum(val_labels)))
            #count number of tumors
            #box, label, count = cv.detect_common_objects(tumor_volume)

            #_, thresh = cv2.threshold(val_labels, 0, 1, cv2.THRESH_BINARY_INV)
            #kernal = np.ones((2, 2), np.uint8)
            #dilation = cv2.dilate(thresh, kernal, iterations=2)
            #contours, hierarchy = cv2.findContours(
            #    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #objects = str(len(contours))
            print("number of tumors",count_objects(val_labels[0,:,:,:]))

            #print("tumor volume: {}".format(tumor_volume))
            dice_list_sub = []
            for i in range(1, 2):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("ImageName, Mean Organ Dice, and Tumor Volume: {}, {}, {}".format(img_name,mean_dice,tumor_volume))
            dice_list_case.append(mean_dice)
            val_inputs_PET = val_inputs.cpu().numpy()[0, 0, :, :, :]
            val_inputs_CT = val_inputs.cpu().numpy()[0, 1, :, :, :]

            val_labels = nib.Nifti1Image(val_labels[0, :, :, :], np.eye(4))  # Save axis for data (just identity)
            # val_labels.header.get_xyzt_units()
            # print(''.join([pretrained_dir,'test/', img_prefix, '_GT.nii.gz']))
            val_labels.to_filename(''.join([pretrained_dir, 'test/', img_prefix, '_GT.nii.gz']))

            val_outputs = nib.Nifti1Image(val_outputs[0, :, :, :], np.eye(4))  # Save axis for data (just identity)
            # val_outputs.header.get_xyzt_units()
            val_outputs.to_filename(
                ''.join([pretrained_dir, 'test/', img_prefix, '_', str(round(mean_dice, 2)), '_Pred.nii.gz']))

            val_inputs_PET = nib.Nifti1Image(val_inputs_PET, np.eye(4))  # Save axis for data (just identity)
            # val_inputs_PET.header.get_xyzt_units()
            val_inputs_PET.to_filename(''.join([pretrained_dir, 'test/', img_prefix, '_PET.nii.gz']))

            val_inputs_CT = nib.Nifti1Image(val_inputs_CT, np.eye(4))  # Save axis for data (just identity)
            # val_inputs_CT.header.get_xyzt_units()
            val_inputs_CT.to_filename(''.join([pretrained_dir, 'test/', img_prefix, '_CT.nii.gz']))

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

if __name__ == '__main__':
    main()