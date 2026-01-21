import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.layer import MeshRenderer
from nets.module import HumanGaussian, SMPLXParamDict, GaussianRenderer
from nets.loss import GeoLoss, RGBLoss, SSIM, LPIPS, LaplacianReg, JointOffsetSymmetricReg, HandMeanReg
from utils.flame import flame
from utils.smpl_x import smpl_x
import copy
from config import cfg
import cv2
import numpy as np
import os
from pytorch3d.structures import Meshes


class Model(nn.Module):
    def __init__(self, human_gaussian, smplx_param_dict):
        super(Model, self).__init__()
        self.human_gaussian = human_gaussian
        self.smplx_param_dict = smplx_param_dict
        self.gaussian_renderer = GaussianRenderer()
        self.face_mesh_renderer = MeshRenderer(flame.vertex_uv, flame.face_uv)
        self.optimizable_params = self.human_gaussian.get_optimizable_params() # for X-Humans dataset, we do not optimize smplx paraeters as it gives better results
        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender])
        self.geo_loss = GeoLoss() 
        self.rgb_loss = RGBLoss()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        self.lap_reg = LaplacianReg(smpl_x.vertex_num_upsampled, smpl_x.face_upsampled)
        self.joint_offset_sym_reg = JointOffsetSymmetricReg()
        self.hand_mean_reg = HandMeanReg()
        self.eval_modules = [self.lpips]
    
    def get_smplx_outputs(self, smplx_param):
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(1,(len(smpl_x.joint_part['body'])-1)*3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(1,len(smpl_x.joint_part['lhand'])*3)
        rhand_pose = smplx_param['rhand_pose'].view(1,len(smpl_x.joint_part['rhand'])*3)
        expr = smplx_param['expr'].view(1,smpl_x.expr_param_dim)
        trans = smplx_param['trans'].view(1,3)
        shape = self.human_gaussian.shape_param[None]
        face_offset = smpl_x.face_offset.cuda()[None]
        joint_offset = self.human_gaussian.joint_offset[None]
        
        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=shape, transl=trans, face_offset=face_offset, joint_offset=joint_offset)
        mesh = output.vertices[0]
        return mesh
    
    def render_normal(self, asset, cam_param, render_shape):
        # world -> camera
        xyz = torch.matmul(cam_param['R'], asset['mean_3d'].permute(1,0)).permute(1,0) + cam_param['t'].view(1,3)

        normal = Meshes(
            verts=xyz[None], 
            faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None]
        ).verts_normals_packed().reshape(smpl_x.vertex_num_upsampled, 3)

        normal = torch.stack((normal[:,0], -normal[:,1], -normal[:,2]), 1)  # flip y,z

        asset_normal = {k: normal if k == 'rgb' else v for k,v in asset.items()}
        normal = self.gaussian_renderer(
            asset_normal, render_shape, cam_param, 
            bg=torch.ones((3)).float().cuda() * -1
        )['img']
        normal = (normal + 1) / 2.  # [0,1]
        return normal
    
    # def forward(self, data, mode):
    # def forward(self, data, mode, cur_itr):
    def forward(self, data, mode, epoch):
        """
        前向传播函数，处理输入数据并返回训练损失或推理输出
        
        Args:
            data: 输入数据字典，包含图像、相机参数、边界框等信息
            mode: 运行模式，'train' 或 其他（如 'test', 'val'）
        
        Returns:
            训练模式下返回损失字典，推理模式下返回输出字典
        """
        batch_size, _, img_height, img_width = data['img'].shape  # 获取输入图像的批次大小和尺寸

        if mode == 'train': # 设置背景颜色：训练时使用随机颜色，推理时使用白色
            bg = torch.rand(3).float().cuda() # 随机RGB颜色
        else:
            bg = torch.ones((3)).float().cuda()  # 白色背景

        # get assets for the rendering and render  # 初始化各种存储容器
        human_assets, human_assets_refined, human_offsets, smplx_outputs = {}, {}, {}, []
        human_renders, human_renders_refined = {}, {}
        face_renders, face_renders_refined = [], []
        for i in range(batch_size):  # 遍历批次中的每个样本
            # 1. 获取SMPLX参数并生成人体高斯资产
            # get assets and offsets from human Gaussians 
            smplx_param = self.smplx_param_dict([data['capture_id'][i]], [data['frame_idx'][i]])[0]
            human_asset, human_asset_refined, human_offset, mesh_neutral_pose = self.human_gaussian(smplx_param)
            # 2. 在训练预热阶段限制高斯尺度，避免内存爆炸
            # clamp scale in early of the training as garbace large scales from randomly initialized networks take HUGE GPU memory
            key_list = ['mean_3d', 'scale', 'rotation', 'rgb']   # 需要处理的关键字段
            if (mode == 'train') and cfg.is_warmup:
                human_asset['scale_wo_clamp'] = human_asset['scale'].clone() # 保存未裁剪的尺度用于后续计算
                human_asset['scale'] = torch.clamp(human_asset['scale'], max=0.001)  # 限制最大尺度
                human_asset_refined['scale_wo_clamp'] = human_asset_refined['scale'].clone()
                human_asset_refined['scale'] = torch.clamp(human_asset_refined['scale'], max=0.001)
                key_list += ['scale_wo_clamp']  # 添加未裁剪尺度到关键字段列表
            # 3. 收集基础资产（原始版本）
            # gather assets
            for key in key_list:
                if key not in human_assets:
                    human_assets[key] = [human_asset[key]]
                    human_assets_refined[key] = [human_asset_refined[key]]
                else:
                    human_assets[key].append(human_asset[key])
                    human_assets_refined[key].append(human_asset_refined[key])
            # 4. 收集偏移量资产
            # gather offsets
            for key in ['mean_offset', 'mean_offset_offset', 'scale_offset', 'rgb_offset']:
                if key not in human_offsets:
                    human_offsets[key] = [human_offset[key]]
                else:
                    human_offsets[key].append(human_offset[key])
            # 5. 获取SMPLX模型输出（网格、关节等）
            # smplx outputs
            smplx_output = self.get_smplx_outputs(smplx_param)
            smplx_outputs.append(smplx_output)
            # 6. 渲染原始版本的人体
            # human render
            human_render = self.gaussian_renderer(human_asset, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()}, bg)
            human_render['normalmap'] = self.render_normal(human_asset, {k: v[i] for k,v in data['cam_param'].items()}, (img_height, img_width))
            
            for key in ['img', 'mask']:
                if key not in human_renders:
                    human_renders[key] = [human_render[key]]
                else:
                    human_renders[key].append(human_render[key])
            # 7. 渲染精炼版本的人体
            # human render (refined)
            human_render_refined = self.gaussian_renderer(human_asset_refined, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()}, bg)
            human_render_refined['normalmap'] = self.render_normal(human_asset, {k: v[i] for k,v in data['cam_param'].items()}, (img_height, img_width))

            for key in ['img', 'mask']:
                if key not in human_renders_refined:
                    human_renders_refined[key] = [human_render_refined[key]]
                else:
                    human_renders_refined[key].append(human_render_refined[key])
            # 8. 渲染面部（使用FLAME模型）
            # face render 
            face_texture, face_texture_mask = flame.texture[None], flame.texture_mask[None,0:1] # 准备面部纹理和遮罩
            face_texture = torch.cat((face_texture, face_texture_mask),1)  # 合并纹理和遮罩
            face_render = self.face_mesh_renderer(face_texture, human_asset['mean_3d'][None,smpl_x.face_vertex_idx,:], flame.face, {k: v[i,None] for k,v in data['cam_param'].items()}, (img_height, img_width))  # 渲染原始版本的面部
            face_render_refined = self.face_mesh_renderer(face_texture, human_asset_refined['mean_3d'][None,smpl_x.face_vertex_idx,:], flame.face, {k: v[i,None] for k,v in data['cam_param'].items()}, (img_height, img_width))  # 渲染精炼版本的面部
            face_renders.append(face_render[0])
            face_renders_refined.append(face_render_refined[0])
        # 9. 聚合所有批次数据（将列表转换为张量）
        # aggregate assets and renders
        human_assets = {k: torch.stack(v) for k,v in human_assets.items()}
        human_assets_refined = {k: torch.stack(v) for k,v in human_assets_refined.items()}
        human_offsets = {k: torch.stack(v) for k,v in human_offsets.items()}
        smplx_outputs = torch.stack(smplx_outputs)
        human_renders = {k: torch.stack(v) for k,v in human_renders.items()}
        human_renders_refined = {k: torch.stack(v) for k,v in human_renders_refined.items()}
        face_renders = torch.stack(face_renders)
        face_renders_refined = torch.stack(face_renders_refined)
        # 10. 训练模式：计算各种损失
        if mode == 'train':
            # loss functions
            loss = {} # 初始化损失字典

            #处理depthmap
            depth_tgt = data['depthmap'].clone().cuda()
            depth_out = human_render['depthmap'][None, None, :, :].clone().cuda()
            depth_out_refined = human_render_refined['depthmap'][None, None, :, :].clone().cuda()
            mask_depth_bool = data['mask'].bool()
            depth_tgt[~mask_depth_bool] = depth_tgt[mask_depth_bool].mean()
            depth_out[~mask_depth_bool] = depth_out[mask_depth_bool].mean().detach()
            depth_out_refined[~mask_depth_bool] = depth_out_refined[mask_depth_bool].mean().detach()

            
            loss['geo'] = self.geo_loss(depth_out, human_render['normalmap'], depth_tgt, data['normalmap']) 
            loss['geo_refined'] = self.geo_loss(depth_out_refined, human_render_refined['normalmap'], depth_tgt, data['normalmap']) 

            
            loss['rgb'] = self.rgb_loss(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight
            loss['ssim'] = (1 - self.ssim(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None])) * cfg.ssim_loss_weight
            loss['lpips'] = self.lpips(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.lpips_weight
            
            is_face = ((face_renders[:,:3] != -1) * (face_renders[:,3:] == 1)).float()
            loss['rgb_face'] = self.rgb_loss(human_renders['img'] * (1 - is_face) + face_renders[:,:3] * is_face, data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight
            
            loss['rgb_refined'] = self.rgb_loss(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight
            loss['ssim_refined'] = (1 - self.ssim(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None])) * cfg.ssim_loss_weight
            loss['lpips_refined'] = self.lpips(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.lpips_weight
            is_face = ((face_renders_refined[:,:3] != -1) * (face_renders_refined[:,3:] == 1)).float()
            loss['rgb_face_refined'] = self.rgb_loss(human_renders_refined['img'] * (1 - is_face) + face_renders_refined[:,:3] * is_face, data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight
           
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda() * 10
            weight[:,self.human_gaussian.is_rhand,:] = 1000 # 右手高权重
            weight[:,self.human_gaussian.is_lhand,:] = 1000 # 左手高权重
            weight[:,self.human_gaussian.is_face,:] = 1  # 面部低权重
            weight[:,self.human_gaussian.is_face_expr,:] = 10 # 表情区域中等权重
            # 均值偏移正则化
            loss['gaussian_mean_reg'] = (human_offsets['mean_offset'] ** 2 + human_offsets['mean_offset_offset'] ** 2) * weight
            # 手部均值特殊正则化
            loss['gaussian_mean_hand_reg'] = self.hand_mean_reg(mesh_neutral_pose, human_offsets['mean_offset'], self.human_gaussian.is_lhand, self.human_gaussian.is_rhand) + self.hand_mean_reg(mesh_neutral_pose, human_offsets['mean_offset_offset'], self.human_gaussian.is_lhand, self.human_gaussian.is_rhand)
            # 尺度正则化权重设置
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda()
            weight[:,self.human_gaussian.is_rhand,:] = 1000 
            weight[:,self.human_gaussian.is_lhand,:] = 1000
            weight[:,self.human_gaussian.is_face_expr,:] = 10
            weight[:,self.human_gaussian.is_cavity,:] = 0  # 空腔区域不惩罚
            # 尺度正则化（区分预热阶段和正常阶段）
            if cfg.is_warmup:
                loss['gaussian_scale_reg'] = (human_assets['scale_wo_clamp'] ** 2 + human_offsets['scale_offset'] ** 2) * weight
            else:
                loss['gaussian_scale_reg'] = (human_assets['scale'] ** 2 + human_offsets['scale_offset'] ** 2) * weight
            # 12. 拉普拉斯平滑正则化（保持网格平滑）
            # 均值拉普拉斯正则化
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda()
            weight[:,self.human_gaussian.is_face_expr,:] = 50 # 表情区域高权重
            weight[:,self.human_gaussian.is_cavity,:] = 0.1 # 空腔区域低权重
            loss['lap_mean'] = (self.lap_reg(mesh_neutral_pose[None,:,:].detach() + human_offsets['mean_offset'], mesh_neutral_pose[None,:,:].detach()) + \
                                self.lap_reg(mesh_neutral_pose[None,:,:].detach() + human_offsets['mean_offset'] + human_offsets['mean_offset_offset'], mesh_neutral_pose[None,:,:].detach())) * 100000 * weight
            # 尺度拉普拉斯正则化
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda()
            weight[:,self.human_gaussian.is_cavity,:] = 0.1
            loss['lap_scale'] = (self.lap_reg(human_assets['scale'], None) + self.lap_reg(human_assets_refined['scale'], None)) * 100000 * weight
            # 关节偏移正则化
            weight = torch.ones((smpl_x.joint_num,3)).float().cuda()
            weight[smpl_x.joint_part['lhand'],:] = 10 # 左手关节高权重
            weight[smpl_x.joint_part['rhand'],:] = 10 # 右手关节高权重
            loss['joint_offset_reg'] = (self.human_gaussian.joint_offset - smpl_x.joint_offset.cuda()) ** 2 * weight
            # 关节偏移对称性正则化
            loss['joint_offset_sym_reg'] = self.joint_offset_sym_reg(self.human_gaussian.joint_offset)
            return loss # 返回所有损失
        else: 
            out = {}
            out['human_depthmap_refined'] = human_render_refined['depthmap'][None, None, :, :]
            out['human_normalmap_refined'] = human_render_refined['normalmap'][None, :, :]
            out['human_img'] = human_renders['img'] # 原始人体渲染
            out['human_img_refined'] = human_renders_refined['img'] # 精炼人体渲染
            out['human_mask_refined'] = human_renders_refined['mask']
            out['smplx_mesh'] = smplx_outputs # SMPLX网格输出
            # 结合人体和面部渲染的最终结果
            is_face = (face_renders[:,:3] != -1).float() * face_renders[:,3:]
            out['human_face_img'] = human_renders['img'] * (1 - is_face) + face_renders[:,:3] * is_face
            is_face = (face_renders_refined[:,:3] != -1).float() * face_renders_refined[:,3:]
            out['human_face_img_refined'] = human_renders_refined['img'] * (1 - is_face) + face_renders_refined[:,:3] * is_face
            return out 
    
def get_model(smplx_params):
    """
    创建并初始化人体高斯渲染模型
    
    Args:
        smplx_params: SMPLX模型参数，如果为None则不初始化SMPLX参数字典
    
    Returns:
        model: 初始化完成的完整模型
    """
    # 1. 创建人体高斯模型实例
    human_gaussian = HumanGaussian()
    with torch.no_grad(): # 使用torch.no_grad()确保初始化过程不计算梯度，节省内存
        human_gaussian.init()  # 初始化人体高斯模型的参数
    # 2. 创建SMPLX参数字典（如果提供了SMPLX参数）
    if smplx_params is not None: 
        smplx_param_dict = SMPLXParamDict() 
        with torch.no_grad(): # 同样使用no_grad()进行初始化
            smplx_param_dict.init(smplx_params) # 使用提供的SMPLX参数初始化
    else:
        smplx_param_dict = None
    # 3. 创建完整的模型，组合人体高斯模型和SMPLX参数字典
    model = Model(human_gaussian, smplx_param_dict) 
    return model
