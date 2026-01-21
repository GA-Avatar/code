import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from config import cfg
from utils.smpl_x import smpl_x
from utils.flame import flame
from utils.preprocessing import load_img, get_bbox
from utils.transforms import transform_joint_to_other_db
import pickle
import json

class XHumans(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.data_split = data_split
        self.root_path = osp.join('..', 'data', 'XHumans', 'data', cfg.subject_id)
        self.transform = transform
        self.img_paths, self.mask_paths, self.depthmap_paths, self.normalmap_paths, self.smplx_params, self.cam_params, self.frame_idx_list = self.load_data()
        self.load_id_info()

    def load_data(self):
        """
        加载数据集，包括图像路径、深度图路径、SMPLX参数、相机参数和帧索引列表
        
        Returns:
            img_paths: 字典，按capture_id和frame_idx组织的图像路径
            depthmap_paths: 字典，按capture_id和frame_idx组织的深度图路径
            smplx_params: 字典，按capture_id和frame_idx组织的SMPLX参数
            cam_params: 字典，按capture_id和frame_idx组织的相机参数
            frame_idx_list: 列表，包含所有有效帧的capture_id和frame_idx信息
        """
        img_paths, mask_paths, depthmap_paths, normalmap_paths, smplx_params, cam_params, frame_idx_list = {}, {}, {}, {}, {}, {}, []  # 初始化空字典和列表来存储各种数据

        capture_path_list = glob(osp.join(self.root_path, self.data_split, '*')) # 获取所有capture文件夹的路径（使用glob查找匹配模式的文件/文件夹）
        for capture_path in capture_path_list:  # 遍历每个capture文件夹
            capture_id = capture_path.split('/')[-1] # 从路径中提取capture_id（最后一个文件夹名）
            
            # load image paths # 1. 加载图像路径
            img_paths[capture_id] = {} 
            img_path_list = glob(osp.join(capture_path, 'render', 'image', '*.png'))
            for img_path in img_path_list:
                frame_idx = int(img_path.split('/')[-1].split('_')[1][:-4])
                img_paths[capture_id][frame_idx] = img_path

            # load mask paths # 2. 加载掩码路径
            mask_paths[capture_id] = {}
            mask_path_list = glob(osp.join(capture_path, 'render', 'masks', '*.png'))
            for mask_path in mask_path_list:
                frame_idx = int(mask_path.split('/')[-1].split('_')[1][:-4])
                mask_paths[capture_id][frame_idx] = mask_path
            
            # load depthmap paths # 2. 加载深度图路径（与图像路径处理方式类似）
            depthmap_paths[capture_id] = {}
            depthmap_path_list = glob(osp.join(capture_path, 'render', 'depths', '*.npy'))
            for depthmap_path in depthmap_path_list:
                frame_idx = int(depthmap_path.split('/')[-1].split('_')[1][:-4])
                depthmap_paths[capture_id][frame_idx] = depthmap_path

            # load normalmap paths # 添加：加载normal路径（与图像路径处理方式类似）
            normalmap_paths[capture_id] = {}
            normalmap_path_list = glob(osp.join(capture_path, 'render', 'normals', '*.npy'))
            for normalmap_path in normalmap_path_list:
                frame_idx = int(normalmap_path.split('/')[-1].split('_')[1][:-4])
                normalmap_paths[capture_id][frame_idx] = normalmap_path

            # load smplx parameters # 3. 加载SMPLX参数
            smplx_params[capture_id] = {}
            smplx_param_path_list = glob(osp.join(capture_path, 'SMPLX', '*.pkl'))
            for smplx_param_path in smplx_param_path_list:
                frame_idx = int(smplx_param_path.split('/')[-1].split('-')[1].split('_')[0][1:])  # 从文件名中提取帧索引（假设格式为"xxx-f123_xxx.pkl"）
                with open(smplx_param_path, 'rb') as f:
                    smplx_param = pickle.load(f, encoding='latin1') # 加载SMPLX参数文件（使用latin1编码处理可能的特殊字符）
                with open(osp.join(capture_path, 'render', 'flame_init', 'flame_params', '%06d.json' % frame_idx)) as f:
                    flame_param = json.load(f) # 加载对应的FLAME参数文件
                if flame_param['is_valid']:  # 如果FLAME参数有效，使用其表情参数；否则使用零向量作为占位符
                    expr = np.array(flame_param['expr'], dtype=np.float32)
                else:
                    expr = np.zeros((flame.expr_param_dim), dtype=np.float32) # dummy  # 使用FLAME库定义的表情参数维度
                smplx_params[capture_id][frame_idx] = {'root_pose': smplx_param['global_orient'], \
                                                        'body_pose': smplx_param['body_pose'].reshape(-1,3), \
                                                        'jaw_pose': smplx_param['jaw_pose'], \
                                                        'leye_pose': smplx_param['leye_pose'], \
                                                        'reye_pose': smplx_param['reye_pose'], \
                                                        'lhand_pose': smplx_param['left_hand_pose'].reshape(-1,3), \
                                                        'rhand_pose': smplx_param['right_hand_pose'].reshape(-1,3), \
                                                        'expr': expr, # use flame's one
                                                        'trans': smplx_param['transl']}  # 组织SMPLX参数，包括各种姿态、表情和平移参数
                smplx_params[capture_id][frame_idx] = {k: torch.FloatTensor(v) for k,v in smplx_params[capture_id][frame_idx].items()} # 将所有参数转换为PyTorch FloatTensor类型

            # load cameras # 4. 加载相机参数
            cam_params[capture_id] = {} 
            cam_param = dict(np.load(osp.join(capture_path, 'render', 'cameras.npz'), allow_pickle=True)) # 加载相机参数NPZ文件（允许pickle以处理可能的复杂数据结构）
            focal = np.array([cam_param['intrinsic'][0][0], cam_param['intrinsic'][1][1]], dtype=np.float32)  # 提取相机内参：焦距和主点
            princpt = np.array([cam_param['intrinsic'][0][2], cam_param['intrinsic'][1][2]], dtype=np.float32)
            R, t = cam_param['extrinsic'][:,:3,:3].astype(np.float32), cam_param['extrinsic'][:,:3,3].astype(np.float32)  # 提取相机外参：旋转矩阵和平移向量
            assert len(R) == len(t)  # 验证外参数组长度与图像数量一致
            assert len(R) == len(img_paths[capture_id])
            for i, frame_idx in enumerate(sorted(list(img_paths[capture_id].keys()))):  # 为每一帧分配相机参数（按帧索引排序）
                cam_params[capture_id][frame_idx] = {'focal': focal, 'princpt': princpt, 'R': R[i], 't': t[i]}
           
            # make frame index  # 5. 创建帧索引列表
            for frame_idx in img_paths[capture_id].keys():
                frame_idx_list.append({'capture_id': capture_id, 'frame_idx': frame_idx})

        return img_paths, mask_paths, depthmap_paths, normalmap_paths, smplx_params, cam_params, frame_idx_list   # 返回所有加载的数据
    
    def load_id_info(self):
        with open(osp.join(self.root_path, 'smplx_optimized', 'shape_param.json')) as f:
            shape_param = torch.FloatTensor(json.load(f))
        with open(osp.join(self.root_path, 'smplx_optimized', 'face_offset.json')) as f:
            face_offset = torch.FloatTensor(json.load(f))
        with open(osp.join(self.root_path, 'smplx_optimized', 'joint_offset.json')) as f:
            joint_offset = torch.FloatTensor(json.load(f))
        with open(osp.join(self.root_path, 'smplx_optimized', 'locator_offset.json')) as f:
            locator_offset = torch.FloatTensor(json.load(f))
        smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

        texture_path = osp.join(self.root_path, 'smplx_optimized', 'face_texture.png')
        texture = torch.FloatTensor(cv2.imread(texture_path)[:,:,::-1].copy().transpose(2,0,1))/255
        texture_mask_path = osp.join(self.root_path, 'smplx_optimized', 'face_texture_mask.png')
        texture_mask = torch.FloatTensor(cv2.imread(texture_mask_path).transpose(2,0,1))/255
        flame.set_texture(texture, texture_mask)

    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        capture_id, frame_idx = self.frame_idx_list[idx]['capture_id'], self.frame_idx_list[idx]['frame_idx']

        # load image
        img = load_img(self.img_paths[capture_id][frame_idx])
        img = self.transform(img.astype(np.float32))/255.

        # # get mask from depthmap
        # depthmap = cv2.imread(self.depthmap_paths[capture_id][frame_idx], -1)[:,:,None]
        # mask = (depthmap < depthmap.max()).astype(np.float32) # 0: bkg, 1: human
        # y, x = np.where(mask[:,:,0])
        # bbox = get_bbox(np.stack((x,y),1), np.ones_like(x))
        # mask = self.transform(mask.astype(np.float32))[0,None,:,:]
        # depthmap = torch.from_numpy(depthmap).float().permute(2,0,1) * mask

        # load mask
        mask = cv2.imread(self.mask_paths[capture_id][frame_idx])[:,:,0,None] / 255.
        y, x = np.where(mask[:,:,0])
        bbox = get_bbox(np.stack((x,y),1), np.ones_like(x))
        mask = self.transform((mask > 0.5).astype(np.float32))

        # load depth
        depth = np.load(self.depthmap_paths[capture_id][frame_idx])[:,:,None]
        depth_mask = np.load(self.depthmap_paths[capture_id][frame_idx][:-4] + '_mask.npy')[:,:,None]
        depthmap = (depth * depth_mask).transpose(2,0,1) # 1, img_height, img_width

        # load normalmap
        normal = np.load(self.normalmap_paths[capture_id][frame_idx]) # [-1,1]
        normal = (normal + 1)/2. # [0,1]
        normal_mask = np.load(self.normalmap_paths[capture_id][frame_idx][:-4] + '_mask.npy')[:,:,None]
        normalmap = (normal * normal_mask).transpose(2,0,1) # 3, img_height, img_width


        data = {'img': img, 'mask': mask,'depthmap': depthmap, 'normalmap': normalmap, 'bbox': bbox, 'cam_param': self.cam_params[capture_id][frame_idx], 'capture_id': capture_id, 'frame_idx': frame_idx}
        return data
