# ------------------------------------------------------------------------------------
# FSOD-TOPG Codebase (https://github.com/NigelLu/FSOD-TOPG)
# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Originated from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F



from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA #* seems to be a self-defined then released package


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None



def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # TODO Non-CUDA version, needs revision
    # * batch_num, patch_len, n_heads, embedding_dim
    B_, _, n_heads_, C_ = value.shape
    # * batch_num, len_query, n_heads, n_levels_, n_points, embedding_dim
    B_, L_query_, n_heads_, n_levels_, n_points_, C_ = sampling_locations.shape

    # N_, _, M_, D_ = value.shape #* batch_num, patch_len, n_heads, embedding_dim
    # _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    # * reform multi-level feature patch maps in a list
    value_list = value.split(
        [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1  # ? why do this?
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):

        # * B_, H_*W_, n_heads_, C_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(
            1, 2).reshape(B_*n_heads_, C_, H_, W_)

        # * B_, L_query_, n_heads_, n_points_, 2 -> B_, n_heads_, L_query_, n_points_, 2 -> B_*n_heads_, L_query_, n_points_, 2
        sampling_grid_l_ = sampling_grids[:, :,
                                          :, lid_].transpose(1, 2).flatten(0, 1)

        #* B_*n_heads_, C_, L_query_, n_points_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)  # * use bilinear because sometimes the normalized sampling_grid_l_ may not have an integer corresponding location
        sampling_value_list.append(sampling_value_l_)

    # * (B_, L_query_, n_heads_, n_levels_, n_points_) -> (B_, n_heads_, L_query_, n_levels_, n_points_) -> (B_, n_heads_, 1, L_query_, n_levels_*n_points_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        B_*n_heads_, 1, L_query_, L_query_*n_points_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(B_, n_heads_*C_, L_query_)
    return output.transpose(1, 2).contiguous()
