# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .convnext_encoder import ConvNextVisionTower
from .hr_clip_encoder import HRCLIPVisionTower
from .vision_models.eva_vit import EVAVITVisionTower
from .sam_encoder import SAMVisionTower
from .pix2struct_encoder import Pix2StructLargeVisionTower
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from copy import deepcopy
import random
import math
import json
import os
from eagle.utils import gumbel_sigmoid
from .gumbel import GumbleSoftmax
import os
import torch
import numpy
def min_max_normalize(score, a=0.0 ,b=1.0):
    min_val, _ = torch.min(score, dim=0, keepdim=True)
    max_val, _ = torch.max(score, dim=0, keepdim=True)
    normalized_score = (score - min_val) / (max_val - min_val + 1e-6) * (b - a) + a
    return normalized_score


def transform_softiou(metric,x,total_steps):
    t = (x - total_steps/2) / total_steps  
    decay = (numpy.tanh(20 * t) + 1) / 2
    return metric * (1 - decay) + (1-metric) * decay

class SampleMultiBackboneChannelConcatenationVisionTower(nn.Module):
    def __init__(self,
                 vision_tower,
                 args,
                 grid_size=32):
        
        super().__init__()

        self.is_loaded = False
        self.grid_size = grid_size
        self.num_tokens = self.grid_size ** 2
        self.samples = args.mm_vision_sample_num
        
        vision_tower_name_list = vision_tower.split(";")
        self.input_image_size = 1024 # hardcode
        self.load_vision_towers(vision_tower_name_list, args)
        
        self.current_iteration = 0
        self.mask_file = 'mask_log.jsonl'
        self.score_file = 'score_log.jsonl'
        self.num_towers = len(self.vision_towers)
        self.num_samples = min(self.samples, self.num_towers)
        self.sampler="uniform"
        self.USE_GUMBEL_SIGMOID = "True"
        self.exp_name = 'x5_observe'
        # self.router = Router(len(self.vision_towers), self.samples)
        if self.num_samples < self.num_towers:
            self.text_aligner = nn.Linear(768, 1024)
            self.sampler = GumbleSoftmax()
            self.query_rater = nn.Linear(1024, 1)
        print(f"build multi encoder v4... check{self.num_samples}...")

    def load_vision_towers(self, vision_tower_name_list, args):
        self.vision_towers = nn.ModuleList()
        self.vision_aligners = nn.ModuleList()
        self.pretrain_vision_towers = nn.ModuleList()
        
        for name in vision_tower_name_list:
            if name == 'det-1024':
                det_args = deepcopy(args)
                det_args.input_image_size = 1024
                det_args.freeze_vision = False
                det_args.vision_tower_pretrained_from = 'checkpoints/eva02_L_coco_det_sys_o365.pth'
                det_vision_tower = EVAVITVisionTower("eva02-l-16", det_args)     
                det_vision_tower.load_model()
                self.vision_towers.append(det_vision_tower)
                
                # Create a frozen version
                det_args.freeze_vision = True  # Set to freeze
                frozen_det_vision_tower = EVAVITVisionTower("eva02-l-16", det_args)
                frozen_det_vision_tower.load_model()
                self.pretrain_vision_towers.append(frozen_det_vision_tower)

                det_vision_aligner = nn.Linear(det_vision_tower.hidden_size, 1024)
                self.vision_aligners.append(det_vision_aligner)

            elif name == 'convnext-1024':
                ## ConvNeXt
                convnext_args = deepcopy(args)
                convnext_args.freeze_vision = False
                convnext_args.input_image_size = 1024
                convnext_vision_tower = "convnext_xxlarge.clip_laion2b_soup" # hardcode
                convnext_vision_tower = ConvNextVisionTower(convnext_vision_tower, 
                                                                convnext_args)
                convnext_vision_tower.load_model()      
                self.vision_towers.append(convnext_vision_tower)
                
                # Create a frozen version
                convnext_args.freeze_vision = True  # Set to freeze
                frozen_convnext_vision_tower = ConvNextVisionTower("convnext_xxlarge.clip_laion2b_soup", 
                                                                    convnext_args)
                frozen_convnext_vision_tower.load_model()
                self.pretrain_vision_towers.append(frozen_convnext_vision_tower)

                convnext_vision_aligner = nn.Linear(convnext_vision_tower.hidden_size, 1024)
                self.vision_aligners.append(convnext_vision_aligner)

            elif name == "sam-1024":
                sam_args = deepcopy(args)
                sam_args.freeze_vision = False
                sam_args.input_image_size = 1024
                sam_args.add_pixel_shuffle = True
                sam_vision_tower = SAMVisionTower("SAM-L", sam_args)
                sam_vision_tower.load_model()
                self.vision_towers.append(sam_vision_tower)
                
                # Create a frozen version
                sam_args.freeze_vision = True  # Set to freeze
                frozen_sam_vision_tower = SAMVisionTower("SAM-L", sam_args)
                frozen_sam_vision_tower.load_model()
                self.pretrain_vision_towers.append(frozen_sam_vision_tower)

                sam_vision_aligner = nn.Linear(sam_vision_tower.hidden_size, 1024)
                self.vision_aligners.append(sam_vision_aligner)

            elif name == 'pix2struct-1024':
                pix_args = deepcopy(args)
                pix_args.input_image_size = 1024
                pix_args.freeze_vision = False
                pix_args.do_resize = True
                pix_args.de_normalize = True
                pix_vision_tower = Pix2StructLargeVisionTower("pix2struct-large", pix_args)     
                pix_vision_tower.load_model()
                self.vision_towers.append(pix_vision_tower)
                
                # Create a frozen version
                pix_args.freeze_vision = True  # Set to freeze
                frozen_pix_vision_tower = Pix2StructLargeVisionTower("pix2struct-large", pix_args)
                frozen_pix_vision_tower.load_model()
                self.pretrain_vision_towers.append(frozen_pix_vision_tower)

                pix_vision_aligner = nn.Linear(pix_vision_tower.hidden_size, 1024)
                self.vision_aligners.append(pix_vision_aligner)

            elif name == 'clip-448':
                clip_args = deepcopy(args)
                clip_args.input_image_size = 336 # actually 448, will have no effect
                clip_args.freeze_vision = False
                clip_vision_tower = HRCLIPVisionTower("openai/clip-vit-large-patch14-336", clip_args)     
                clip_vision_tower.load_model()
                self.vision_towers.append(clip_vision_tower)
                
                # Create a frozen version
                clip_args.freeze_vision = True  # Set to freeze
                frozen_clip_vision_tower = HRCLIPVisionTower("openai/clip-vit-large-patch14-336", clip_args)
                frozen_clip_vision_tower.load_model()
                self.pretrain_vision_towers.append(frozen_clip_vision_tower)

                clip_vision_aligner = nn.Linear(clip_vision_tower.hidden_size, 1024)
                self.vision_aligners.append(clip_vision_aligner)
        
        # a hardcode here, so we always use convnext in the vision encoder mixture
        self.image_processor = convnext_vision_tower.image_processor
        self.is_loaded = True


        
        
    def load_model(self):
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"

    def forward(self, x, quest_local_features=None, quest_global_features=None, quest_labels = None):
        
        features = []
        masks = []
        attns = []
        for i, (vision_tower, vision_aligner) in enumerate(zip(self.vision_towers, self.vision_aligners)):
            if vision_tower.input_image_size != self.input_image_size:
                resized_x = F.interpolate(x.float(), 
                                          size=(vision_tower.input_image_size, vision_tower.input_image_size), 
                                          mode='bilinear', 
                                          align_corners=True).to(dtype=x.dtype)
            else:
                resized_x = x
            feature = vision_tower(resized_x)
            # 得到预训练视觉编码的图像
            p_feature = self.pretrain_vision_towers[i](resized_x) 
            if len(feature.shape) == 3: # b, n, c
                feature = vision_aligner(feature)
                b, n, c = feature.shape
                b, pn, pc = p_feature.shape
                if n == self.num_tokens:
                    features.append(feature)
                    p_feature = p_feature.transpose(1, 2).reshape(b, pc, self.grid_size, self.grid_size) # (b,n,c)-->(b,c,grid,grid)
                    p_global_feature = F.adaptive_avg_pool2d(p_feature, (1, 1)) # (b,c,grid,grid) --> (b,c)
                    p_global_feature = p_global_feature.expand(-1, -1 , self.grid_size, self.grid_size) # (b,c,1,1) --> (b,n,c)
                    image_sim = F.cosine_similarity(p_feature, p_global_feature, dim =1) # (b, n ,c) vs (b, n, c) -->(b,n,1) 
                    attns.append(image_sim)     
                    continue

                w = h = int(n**0.5)
                feature = feature.transpose(1, 2).reshape(b, c, h, w)
                p_feature = p_feature.transpose(1, 2).reshape(b, pc, h, w)
                p_global_feature = F.adaptive_avg_pool2d(p_feature, (1, 1))
            else:
                b, c, h, w = feature.shape
                b, pc, ph, ph = p_feature.shape
                feature = feature.reshape(b, c, h*w).transpose(-1, -2)
                feature = vision_aligner(feature)
                feature = feature.transpose(-1, -2).reshape(b, -1, h, w)
                p_global_feature = F.adaptive_avg_pool2d(p_feature, (1, 1))

            if w != self.grid_size:
                feature = F.interpolate(feature.float(), size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=True).to(dtype=x.dtype)
                p_feature = F.interpolate(p_feature.float(), size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=True).to(dtype=x.dtype)
            
            p_global_feature = p_global_feature.expand(-1, -1 , self.grid_size, self.grid_size)
            image_sim = F.cosine_similarity(p_global_feature, p_feature, dim = 1) # (b,c,1,1) (b,c,h,w) --> (b,h,w)
            attns.append(image_sim)
            features.append(feature.flatten(2,3).transpose(1,2))  # (B, N, C)

        if self.num_samples < self.num_towers:
            X, B, s, d = quest_local_features.shape[0], feature.shape[0], feature.shape[1],features[0].shape[2]
            quest_global_features = self.text_aligner(quest_global_features)
            quest_local_features = self.text_aligner(quest_local_features)
            
            quest_masks = torch.zeros((X, B), device=x.device, dtype = x.dtype) 
            quest_masks[torch.arange(X), quest_labels] = 1 
            quest_masks = quest_masks.T # (B,X) 
            quest_num = torch.sum(quest_masks, dim = -1) # [B,1]

            quest_global_features = quest_masks @ quest_global_features # [B, X] vs [X,c] --> [B, c]
            quest_global_features = quest_global_features / quest_num.unsqueeze(-1)
            query_rate = self.query_rater(quest_global_features)
            if self.training:
                self.log_masks(query_rate, f'query_rate_{os.getenv("SLURM_NODEID")}.jsonl') 


            quest_local_features = quest_masks.unsqueeze(0) @ quest_local_features.transpose(0, 1) # [1, B, X] vs [X,s,c] --》 [s, B, c]
            quest_local_features = quest_local_features.transpose(0, 1) # [s, B, c] --> [B, s, c]
            quest_local_features = quest_local_features / quest_num.unsqueeze(-1).unsqueeze(-1)
            quest_local_features = F.normalize(quest_local_features, dim = -1)


            batch_features = F.normalize(torch.cat(features, dim=0), dim =-1)  # [4*B,N, c]
            ################################### local question score ###################################            
            quest_local_features = quest_local_features.unsqueeze(0).transpose(-1, -2) # (B,s,c) --> (1, B, s, c) --> (1, B, c, s)
            batch_features = batch_features.reshape(self.num_towers, B, s, -1) # [4,B,N,c]
            local_question_aware = batch_features @ quest_local_features  # (4, B, N, c） (1, B, c, s)-->[4, B, N, s]

            local_question_aware, _  = torch.max(local_question_aware, dim = -2) # [4,B,N,s]-->[4,B,s]
            local_question_aware, _ = torch.max(local_question_aware, dim = -1) # [4,B,s] --> (4,B) (padding问题)感觉还是转成一个question比较合适

            if self.training:
                self.log_masks(local_question_aware.T, f'before_local_question_log_{os.getenv("SLURM_NODEID")}.jsonl') 

            local_question_aware = local_question_aware[1:,:]
            ones_for_clip = torch.ones((1, B), dtype=local_question_aware.dtype, device=local_question_aware.device)
            local_question_aware = torch.cat((ones_for_clip, local_question_aware), dim = 0) # (4,B)
            local_question_aware = min_max_normalize(local_question_aware)

            query_rate = query_rate.T
            local_question_aware = (1 + query_rate) * local_question_aware # [4,B]
            if self.training:
                self.log_masks(local_question_aware.T, f'after_local_question_log_{os.getenv("SLURM_NODEID")}.jsonl') 

            ################################### softiou score ###################################
            clip_map = attns[0].unsqueeze(1) # (B, 1, n,n)
            attns = torch.cat([attn.unsqueeze(1) for attn in attns], dim=1) # (B, 4, n ,n)
            intersection = torch.min(clip_map, attns)
            intersection_sum = intersection.sum(dim=(-1, -2))  # 对 (n, n) 维度求和，结果是形状为 (b,) 的 tensor

            union = torch.max(clip_map, attns)
            union_sum = union.sum(dim=(-1, -2))  # 对 (n, n) 维度求和，结果是形状为 (b,) 的 tensor

            soft_iou = intersection_sum / (union_sum + 1e-6)  # 加小值防止分母为零
            if self.training:
                soft_iou = transform_softiou(soft_iou, self.current_iteration, 14138)[:,1:]
                ones_for_clip = torch.ones((B,1), dtype=soft_iou.dtype, device=soft_iou.device)
                soft_iou = torch.cat((ones_for_clip, soft_iou), dim =-1)
            else:
                soft_iou = transform_softiou(soft_iou, 14138, 14138)[:,1:]
                ones_for_clip = torch.ones((B,1), dtype=soft_iou.dtype, device=soft_iou.device)
                soft_iou = torch.cat((ones_for_clip, soft_iou), dim =-1) # (B,4)
            soft_iou = soft_iou.T 
            if self.training:
                self.log_masks(soft_iou.T, f'soft_iou_log_{os.getenv("SLURM_NODEID")}.jsonl') 
            
            score = local_question_aware + soft_iou
            masks = score.T

            features = torch.cat([feature.unsqueeze(1) for feature in features], dim=1)# [(B,4,N,C)) 
            features = features * masks.unsqueeze(-1).unsqueeze(-1)
            features = features.transpose(1,2).reshape(B, s, -1) # -->[B,N,4,C] -->[B,N,4*c] [4,B,B]
            self.current_iteration += 1
        else:
            features = torch.cat(features, dim=-1)
        
        return features

    def log_masks(self, masks, file_name):
        # if os.path.exists(f'observe/{self.exp_name}') is False:
        #     os.makedirs(f'observe/{self.exp_name}')
        with open(f'observe/{self.exp_name}/{file_name}', 'a') as f:  # Added a comma here
            for mask in masks:
                f.write(json.dumps(mask.to(torch.float32).detach().cpu().numpy().tolist()) + '\n')
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.clip_vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.clip_vision_tower.parameters()).device

    @property
    def config(self):
        assert NotImplementedError
        pass

    #@property
    # def hidden_size(self):
    #     return sum([_.hidden_size for _ in self.vision_towers])
    @property
    def hidden_size(self):
        return 1024 * len(self.vision_towers)


    @property
    def num_patches(self):
        return self.num_tokens