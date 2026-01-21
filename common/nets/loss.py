import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import lpips
from utils.smpl_x import smpl_x
from pytorch3d.structures import Meshes
import cv2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')) # 添加项目根目录到Python路径
from main.config import cfg
# import random
from random import randint 

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
    def pearson_corrcoef(self, x, y): # 计算皮尔逊相关系数 (Pearson correlation coefficient)
        x = x - x.mean() # 输入: x, y 为 1D 向量 (已展平的预测与GT)
        y = y - y.mean()
        return torch.sum(x * y) / (torch.norm(x) * torch.norm(y) + 1e-8)  # 输出: [-1, 1] 之间的相关系数，越接近1相关性越高
    def normalize(self, input, mean=None, std=None): # 对输入做归一化
        input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
        input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
        return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))
    def patchify(self, input, patch_size):  # 将输入划分为 patch
        patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size) # [B, C*P*P, L]→[B, L, C*P*P]→[B*L, P*P]
        return patches
    
    def patch_norm_mse_loss(self, input, target, patch_size, margin, return_mask=False): # Patch 级别的归一化 + L2损失 (局部)
        input_patches = self.normalize(self.patchify(input, patch_size))  # 对每个patch独立归一化
        target_patches = self.normalize(self.patchify(target, patch_size))
        return self.margin_l2_loss(input_patches, target_patches, margin, return_mask)
        # return pearson_correlation_loss(input_patches, target_patches, margin, return_mask)

    def patch_norm_mse_loss_global(self, input, target, patch_size, margin, return_mask=False): # Patch 级别的归一化 + L2损失 (全局)
        input_patches = self.normalize(self.patchify(input, patch_size), std = input.std().detach()) # 用整张图的std来归一化，而不是每个patch单独的std
        target_patches = self.normalize(self.patchify(target, patch_size), std = target.std().detach())
        return self.margin_l2_loss(input_patches, target_patches, margin, return_mask)
        # return pearson_correlation_loss(input_patches, target_patches, margin, return_mask)

    def margin_l2_loss(self, depth_out, depth_tgt, margin, return_mask=False):  # 带有 margin 的 L2 损失
        mask = (depth_out - depth_tgt).abs() > margin # 仅在 (预测 - GT) > margin 的地方计算平方误差
        if not return_mask:
            return ((depth_out - depth_tgt)[mask] ** 2).mean()
        else:
            return ((depth_out - depth_tgt)[mask] ** 2).mean(), mask

    def forward(self, depth_out, depth_tgt, patch_size, margin=0.02):
        local_loss = self.patch_norm_mse_loss(depth_out, depth_tgt, patch_size, margin)
        global_loss = self.patch_norm_mse_loss_global(depth_out, depth_tgt, patch_size, margin)
        
        # 调用皮尔逊相关系数
        pred_flat = depth_out.view(-1)
        gt_flat = depth_tgt.view(-1)
        pearson_loss = 1.0 - self.pearson_corrcoef(pred_flat, gt_flat)
        
        total_loss = cfg.depth_local_loss_weight * local_loss + cfg.depth_global_loss_weight * global_loss + cfg.depth_pearson_loss_weight * pearson_loss
        return total_loss
# class DepthLoss(nn.Module):
#     def __init__(self):
#         super(DepthLoss, self).__init__()

#     # 皮尔逊相关系数
#     def pearson_corrcoef(self, x, y):
#         x = x - x.mean()
#         y = y - y.mean()
#         denom = (torch.norm(x) * torch.norm(y) + 1e-8)
#         if denom == 0:  # 防止除0
#             return torch.tensor(0.0, device=x.device)
#         return torch.sum(x * y) / denom

#     # 标准化，避免小std放大噪声
#     def normalize(self, input, mean=None, std=None):
#         input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
#         input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
#         global_std = torch.std(input.reshape(-1))
#         return (input - input_mean) / (input_std + 1e-2 * global_std + 1e-6)

#     # 划分 patch
#     def patchify(self, input, patch_size):
#         B, C, H, W = input.shape
#         patches = F.unfold(input, kernel_size=patch_size, stride=patch_size)  # [B, C*P*P, L]
#         patches = patches.permute(0, 2, 1)  # [B, L, C*P*P]
#         patches = patches.reshape(-1, C * patch_size * patch_size)  # [B*L, C*P*P]
#         return patches

#     # patch 局部损失
#     def patch_norm_mse_loss(self, input, target, patch_size, margin, return_mask=False):
#         input_patches = self.normalize(self.patchify(input, patch_size))
#         target_patches = self.normalize(self.patchify(target, patch_size))
#         return self.margin_l2_loss(input_patches, target_patches, margin, return_mask)

#     # patch 全局损失
#     def patch_norm_mse_loss_global(self, input, target, patch_size, margin, return_mask=False):
#         input_patches = self.normalize(self.patchify(input, patch_size), std=input.std().detach())
#         target_patches = self.normalize(self.patchify(target, patch_size), std=target.std().detach())
#         return self.margin_l2_loss(input_patches, target_patches, margin, return_mask)

#     # 带 margin 的 L2 损失
#     def margin_l2_loss(self, pred, gt, margin, return_mask=False):
#         mask = (pred - gt).abs() > margin
#         if mask.sum() == 0:  # 防止空mask
#             loss = torch.tensor(0.0, device=pred.device)
#         else:
#             loss = ((pred - gt)[mask] ** 2).mean()
#         return (loss, mask) if return_mask else loss

#     def forward(self, depth_out, depth_tgt, patch_size, margin=0.02):
#         # -------- 前景 mask --------
#         valid_mask = (depth_tgt > 0).float()

#         # 局部 / 全局 patch loss
#         local_loss = self.patch_norm_mse_loss(depth_out * valid_mask, depth_tgt * valid_mask, patch_size, margin)
#         global_loss = self.patch_norm_mse_loss_global(depth_out * valid_mask, depth_tgt * valid_mask, patch_size, margin)

#         # Pearson loss 仅在前景区域计算
#         pred_flat = depth_out[valid_mask > 0].view(-1)
#         gt_flat = depth_tgt[valid_mask > 0].view(-1)
#         if pred_flat.numel() == 0:  # 没有前景
#             pearson_loss = torch.tensor(0.0, device=depth_out.device)
#         else:
#             pearson_loss = 1.0 - self.pearson_corrcoef(pred_flat, gt_flat)

#         # # 背景正则: 约束背景接近0
#         # bg_loss = ((depth_out[depth_tgt == 0]) ** 2).mean() if (depth_tgt == 0).any() else 0.0

#         # # 总损失
#         # total_loss = (
#         #     self.local_w * local_loss +
#         #     self.global_w * global_loss +
#         #     self.pearson_w * pearson_loss +
#         #     self.bg_w * bg_loss
#         # )
#         # 总损失
#         total_loss = cfg.depth_local_loss_weight * local_loss + cfg.depth_global_loss_weight * global_loss + cfg.depth_pearson_loss_weight * pearson_loss
#         return total_loss

# class DepthLoss(nn.Module):
#     def __init__(self):
#         super(DepthLoss, self).__init__()

#     def forward(self, depth_out, depth_tgt):
#         batch_size = depth_out.shape[0]

#         loss = 0
#         for i in range(batch_size):
#             # get foreground
#             is_valid = (depth_out[i] > 0) * (depth_tgt[i] > 0) # 创建一个 mask，标记深度值有效的像素（大于0）, 只有预测深度和真实深度都大于0的像素才参与损失计算
#             depth_out_i = depth_out[i][is_valid] # 根据 mask 提取有效像素，变成一维张量
#             depth_tgt_i = depth_tgt[i][is_valid]

#             # normalize in [0,1]
#             depth_out_normalized = (depth_out_i - torch.min(depth_out_i)) / (torch.max(depth_out_i) - torch.min(depth_out_i) + 1e-4) # 将有效深度像素归一化到 [0,1] 范围
#             depth_tgt_normalized = (depth_tgt_i - torch.min(depth_tgt_i)) / (torch.max(depth_tgt_i) - torch.min(depth_tgt_i) + 1e-4)  # 避免不同图像深度值范围差异过大,1e-4 防止除零错误
#             is_valid = (depth_out_normalized > 0) * (depth_tgt_normalized > 0) # 再次筛选归一化后大于0的像素，防止归一化导致的数值异常
#             depth_out_normalized, depth_tgt_normalized = depth_out_normalized[is_valid], depth_tgt_normalized[is_valid] # 对归一化深度取对数，然后计算预测与真实的差值

#             # compute loss
#             diff_log = torch.log(depth_tgt_normalized) - torch.log(depth_out_normalized) # 计算差值的均值，用于消除整体偏差
#             diff_log_mean = torch.mean(diff_log)  # 计算差值平方的均值（MSE 风格），表示误差强度
#             diff_log_sq_mean = torch.mean(diff_log**2)  # 计算每张图像的“对数深度误差（log-depth loss）”
#             loss += torch.sqrt(diff_log_sq_mean - 0.5*diff_log_mean**2)
#         loss = loss / batch_size
#         return loss
    
class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, normal_out, normal_tgt):
        is_valid_out = 1 - ((normal_out==0).sum(1)[:,None] == 3).float()
        is_valid_tgt = 1 - ((normal_tgt==0).sum(1)[:,None] == 3).float()
        normal_out = normal_out*2-1 # [0,1] -> [-1,1]
        normal_tgt = normal_tgt*2-1 # [0,1] -> [-1,1]
        loss = (torch.abs(normal_out - normal_tgt) + (1 - torch.sum(normal_out*normal_tgt,1)[:,None])) * is_valid_out * is_valid_tgt
        return loss
    
class GeoLoss(nn.Module):
    def __init__(self):
        super(GeoLoss, self).__init__()
        self.depth_loss = DepthLoss()
        self.normal_loss = NormalLoss()

    def forward(self, depth_out, normal_out, depth_tgt, normal_tgt):
        #patch-depth
        patch_range = (5, 17)
        depth_loss = self.depth_loss(depth_out, depth_tgt, randint(patch_range[0], patch_range[1]), 0.02) * cfg.depth_loss_weight

        # depth_loss = self.depth_loss(depth_out, depth_tgt) * cfg.depth_loss_weight
        normal_loss = self.normal_loss(normal_out, normal_tgt) * cfg.normal_loss_weight
        loss = depth_loss + normal_loss 
        # loss = depth_loss
        # loss = normal_loss
        return loss
        
class RGBLoss(nn.Module):
    def __init__(self):
        super(RGBLoss, self).__init__()
    
    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:,:,None,None]
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]

        loss = torch.abs(img_out - img_target)
        return loss

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.FloatTensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).cuda()
        return gauss / gauss.sum()

    def create_window(self, window_size, feat_dim):
        window_1d = self.gaussian(window_size, 1.5)[:,None]
        window_2d = torch.mm(window_1d, window_1d.permute(1,0))[None,None,:,:]
        window_2d = window_2d.repeat(feat_dim,1,1,1)
        return window_2d

    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None, window_size=11):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:,:,None,None]
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]

        window = self.create_window(window_size, feat_dim)
        mu1 = F.conv2d(img_out, window, padding=window_size//2, groups=feat_dim)
        mu2 = F.conv2d(img_target, window, padding=window_size//2, groups=feat_dim)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img_out*img_out, window, padding=window_size//2, groups=feat_dim) - mu1_sq
        sigma2_sq = F.conv2d(img_target*img_target, window, padding=window_size//2, groups=feat_dim) - mu2_sq
        sigma1_sigma2 = F.conv2d(img_out*img_target, window, padding=window_size//2, groups=feat_dim) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1_sigma2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map

# image perceptual loss (LPIPS. https://github.com/richzhang/PerceptualSimilarity)
class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.lpips = lpips.LPIPS(net='vgg').cuda()

    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:,:,None,None]
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]
        img_out = img_out * 2 - 1 # [0,1] -> [-1,1]
        img_target = img_target * 2 - 1 # [0,1] -> [-1,1]
        loss = self.lpips(img_out, img_target)
        return loss

class LaplacianReg(nn.Module):
    def __init__(self, vertex_num, face):
        super(LaplacianReg, self).__init__()
        self.neighbor_idxs, self.neighbor_weights = self.get_neighbor(vertex_num, face)

    def get_neighbor(self, vertex_num, face, neighbor_max_num = 10):
        adj = {i: set() for i in range(vertex_num)}
        for i in range(len(face)):
            for idx in face[i]:
                adj[idx] |= set(face[i]) - set([idx])

        neighbor_idxs = np.tile(np.arange(vertex_num)[:,None], (1, neighbor_max_num))
        neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)
        for idx in range(vertex_num):
            neighbor_num = min(len(adj[idx]), neighbor_max_num)
            neighbor_idxs[idx,:neighbor_num] = np.array(list(adj[idx]))[:neighbor_num]
            neighbor_weights[idx,:neighbor_num] = -1.0 / neighbor_num
        
        neighbor_idxs, neighbor_weights = torch.from_numpy(neighbor_idxs).cuda(), torch.from_numpy(neighbor_weights).cuda()
        return neighbor_idxs, neighbor_weights
    
    def compute_laplacian(self, x, neighbor_idxs, neighbor_weights):
        lap = x + (x[:, neighbor_idxs] * neighbor_weights[None, :, :, None]).sum(2)
        return lap

    def forward(self, out, target):
        if target is None:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            loss = lap_out ** 2
            return loss
        else:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            lap_target = self.compute_laplacian(target, self.neighbor_idxs, self.neighbor_weights)
            loss = (lap_out - lap_target) ** 2
            return loss

class JointOffsetSymmetricReg(nn.Module):
    def __init__(self):
        super(JointOffsetSymmetricReg, self).__init__()
    
    def forward(self, joint_offset):
        right_joint_idx, left_joint_idx = [], []
        for j in range(smpl_x.joint_num):
            if smpl_x.joints_name[j][:2] == 'R_':
                right_joint_idx.append(j)
                idx = smpl_x.joints_name.index('L_' + smpl_x.joints_name[j][2:])
                left_joint_idx.append(idx)

        loss = torch.abs(joint_offset[right_joint_idx,0] + joint_offset[left_joint_idx,0]) + torch.abs(joint_offset[right_joint_idx,1] - joint_offset[left_joint_idx,1]) + torch.abs(joint_offset[right_joint_idx,2] - joint_offset[left_joint_idx,2])
        return loss

class HandMeanReg(nn.Module):
    def __init__(self):
        super(HandMeanReg, self).__init__()
 
    def forward(self, mesh_neutral_pose, offset, is_rhand, is_lhand):
        batch_size = offset.shape[0]
        is_hand = (is_rhand + is_lhand) > 0
        with torch.no_grad():
            normal = Meshes(verts=mesh_neutral_pose[None,:,:], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None,:,:]).verts_normals_packed().reshape(1,smpl_x.vertex_num_upsampled,3).detach().repeat(batch_size,1,1)
        dot_prod = torch.sum(normal * F.normalize(offset, p=2, dim=2), 2)[:,is_hand]
        loss = torch.clamp(dot_prod, min=0)
        return loss

