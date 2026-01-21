import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from utils.transforms import eval_sh, RGB2SH, get_fov, get_view_matrix, get_proj_matrix
from utils.smpl_x import smpl_x
from smplx.lbs import batch_rigid_transform
#from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from nets.layer import make_linear_layers
from pytorch3d.structures import Meshes
from config import cfg
import copy

class HumanGaussian(nn.Module):
    def __init__(self):
        super(HumanGaussian, self).__init__()
        # 1. 三平面表示参数（用于几何和外观编码） # 三平面表示：三个正交的特征平面（XY、XZ、YZ）
        self.triplane = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())
        self.triplane_face = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())
        # 2. 几何网络：从三平面特征预测几何属性
        self.geo_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128], use_gn=True)
        self.mean_offset_net = make_linear_layers([128, 3], relu_final=False) # 均值偏移
        self.scale_net = make_linear_layers([128, 1], relu_final=False) # 尺度
        # 3. 几何偏移网络：考虑姿态依赖的几何变化 # 输入：三平面特征 + 关节旋转信息 (smpl_x.joint_num-1)*6 (旋转矩阵的6D表示)
        self.geo_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(smpl_x.joint_num-1)*6, 128, 128, 128], use_gn=True)
        self.mean_offset_offset_net = make_linear_layers([128, 3], relu_final=False) # 额外的均值偏移
        self.scale_offset_net = make_linear_layers([128, 1], relu_final=False)  # 额外的尺度偏移
        # 4. 颜色网络：预测RGB颜色
        self.rgb_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128, 3], relu_final=False, use_gn=True)
        # 5. 颜色偏移网络：考虑姿态和视角依赖的颜色变化 # 输入：三平面特征 + 关节旋转 + 视角方向(3D)
        self.rgb_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(smpl_x.joint_num-1)*6+3, 128, 128, 128, 3], relu_final=False, use_gn=True)
        # 6. SMPLX相关参数和层
        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender]).cuda()  # SMPLX模型层
        self.shape_param = nn.Parameter(smpl_x.shape_param.float().cuda()) # 形状参数（可学习）
        self.joint_offset = nn.Parameter(smpl_x.joint_offset.float().cuda())  # 关节偏移（可学习）
     
    def init(self):
        """
        初始化模型，准备SMPLX相关的缓冲区和上采样网格
        """
        # 1. 获取中性姿态的人体网格顶点
        # upsample mesh and other assets
        xyz, _, _, _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=False)
        # xyz, _, _ = self.get_zero_pose_human(return_mesh=True)
        # 2. 获取SMPLX的蒙皮权重和各种变换矩阵
        skinning_weight = self.smplx_layer.lbs_weights.float() # 线性混合蒙皮权重
        pose_dirs = self.smplx_layer.posedirs.permute(1,0).reshape(smpl_x.vertex_num,3*(smpl_x.joint_num-1)*9) # 姿态依赖的形变
        expr_dirs = self.smplx_layer.expr_dirs.view(smpl_x.vertex_num,3*smpl_x.expr_param_dim) # 表情依赖的形变
        # 3. 创建身体部位掩码（右手、左手、面部、表情区域）
        is_rhand, is_lhand, is_face, is_face_expr = torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda()
        is_rhand[smpl_x.rhand_vertex_idx], is_lhand[smpl_x.lhand_vertex_idx], is_face[smpl_x.face_vertex_idx], is_face_expr[smpl_x.expr_vertex_idx] = 1.0, 1.0, 1.0, 1.0 # 设置对应顶点的掩码值
        # 4. 空腔区域掩码（可能是内部不可见区域）
        is_cavity = torch.FloatTensor(smpl_x.is_cavity).cuda()[:,None]
        # 5. 上采样网格和相关属性到更高分辨率 # 使用虚拟顶点进行上采样
        _, skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_cavity = smpl_x.upsample_mesh(torch.ones((smpl_x.vertex_num,3)).float().cuda(), [skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_cavity]) # upsample with dummy vertex
        # 6. 重新组织上采样后的数据
        pose_dirs = pose_dirs.reshape(smpl_x.vertex_num_upsampled*3,(smpl_x.joint_num-1)*9).permute(1,0) 
        expr_dirs = expr_dirs.view(smpl_x.vertex_num_upsampled,3,smpl_x.expr_param_dim)
        is_rhand, is_lhand, is_face, is_face_expr = is_rhand[:,0] > 0, is_lhand[:,0] > 0, is_face[:,0] > 0, is_face_expr[:,0] > 0 # 将掩码转换为布尔值
        is_cavity = is_cavity[:,0] > 0
        # 7. 注册缓冲区（不参与梯度计算但需要保存的常量）
        self.register_buffer('pos_enc_mesh', xyz)  # 位置编码网格
        self.register_buffer('skinning_weight', skinning_weight)  # 蒙皮权重
        self.register_buffer('pose_dirs', pose_dirs) # 姿态依赖形变
        self.register_buffer('expr_dirs', expr_dirs)  # 表情依赖形变
        self.register_buffer('is_rhand', is_rhand) # 右手区域掩码
        self.register_buffer('is_lhand', is_lhand) # 左手区域掩码
        self.register_buffer('is_face', is_face) # 面部区域掩码
        self.register_buffer('is_face_expr', is_face_expr) # 表情区域掩码
        self.register_buffer('is_cavity', is_cavity)  # 空腔区域掩码

    def get_optimizable_params(self):
        optimizable_params = [
            {'params': [self.triplane], 'name': 'triplane_human', 'lr': cfg.lr},
            {'params': [self.triplane_face], 'name': 'triplane_face_human', 'lr': cfg.lr},
            {'params': list(self.geo_net.parameters()), 'name': 'geo_net_human', 'lr': cfg.lr},
            {'params': list(self.mean_offset_net.parameters()), 'name': 'mean_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.scale_net.parameters()), 'name': 'scale_net_human', 'lr': cfg.lr},
            {'params': list(self.geo_offset_net.parameters()), 'name': 'geo_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.mean_offset_offset_net.parameters()), 'name': 'mean_offset_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.scale_offset_net.parameters()), 'name': 'scale_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.rgb_net.parameters()), 'name': 'rgb_net_human', 'lr': cfg.lr},
            {'params': list(self.rgb_offset_net.parameters()), 'name': 'rgb_offset_net_human', 'lr': cfg.lr},
            {'params': [self.shape_param], 'name': 'shape_param_human', 'lr': cfg.lr},
            {'params': [self.joint_offset], 'name': 'joint_offset_human', 'lr': cfg.lr}
        ]
        return optimizable_params

    def get_neutral_pose_human(self, jaw_zero_pose, use_id_info):
        """
        获取中性姿态的人体网格和变换信息
        
        Args:
            jaw_zero_pose: 是否使用零下巴姿态（闭合嘴巴）
            use_id_info: 是否使用身份信息（形状参数和偏移）
        
        Returns:
            mesh_neutral_pose_upsampled: 上采样后的中性姿态网格
            mesh_neutral_pose: 原始分辨率的中性姿态网格
            joint_neutral_pose: 中性姿态的关节位置
            transform_mat_neutral_pose: 从T-pose到零姿态的变换矩阵
        """
        # 1. 初始化各种姿态参数为零或中性值
        zero_pose = torch.zeros((1,3)).float().cuda() # 零旋转（3维轴角）
        neutral_body_pose = smpl_x.neutral_body_pose.view(1,-1).cuda() # 大 pose # 身体中性姿态
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda() # 手部零姿态（放松状态）
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda() # 零表情参数
        # 2. 处理下巴姿态
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()  # 零下巴姿态（闭合嘴巴）
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1,3).cuda() # open mouth  # 中性下巴姿态（微微张开）
        # 3. 处理身份相关信息
        if use_id_info:
            shape_param = self.shape_param[None,:]  # 可学习的形状参数
            face_offset = smpl_x.face_offset[None,:,:].float().cuda()   # 面部顶点偏移
            joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:]) # 关节偏移
            #joint_offset = None
        else:
            shape_param = torch.zeros((1,smpl_x.shape_param_dim)).float().cuda()  # 零形状参数
            face_offset = None # 无面部偏移
            joint_offset = None  # 无关节偏移
        # 4. 通过SMPLX层前向传播获取网格和关节
        output = self.smplx_layer(global_orient=zero_pose, body_pose=neutral_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        # 5. 提取输出结果
        mesh_neutral_pose = output.vertices[0] # 大 pose human # 原始分辨率的网格顶点
        mesh_neutral_pose_upsampled = smpl_x.upsample_mesh(mesh_neutral_pose) # 大 pose human # 上采样到高分辨率
        joint_neutral_pose = output.joints[0][:smpl_x.joint_num,:] # 大 pose human # 关节位置（排除末端效应器）
        # 6. 计算从大pose到零姿态的变换矩阵
        # compute transformation matrix for making 大 pose to zero pose
        neutral_body_pose = neutral_body_pose.view(len(smpl_x.joint_part['body'])-1,3)  # 分解中性身体姿态
        zero_hand_pose = zero_hand_pose.view(len(smpl_x.joint_part['lhand']),3) # 分解手部姿态
        neutral_body_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(neutral_body_pose))) # 计算身体姿态的逆变换（从大-pose回到零姿态）
        jaw_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(jaw_pose))) # 计算下巴姿态的逆变换
        pose = torch.cat((zero_pose, neutral_body_pose_inv, jaw_pose_inv, zero_pose, zero_pose, zero_hand_pose, zero_hand_pose))  # 组合所有关节的变换（按SMPLX关节顺序）
        pose = axis_angle_to_matrix(pose) # 将轴角转换为旋转矩阵
        _, transform_mat_neutral_pose = batch_rigid_transform(pose[None,:,:,:], joint_neutral_pose[None,:,:], self.smplx_layer.parents) # 计算刚性变换矩阵
        transform_mat_neutral_pose = transform_mat_neutral_pose[0]  # 去除批量维度
        return mesh_neutral_pose_upsampled, mesh_neutral_pose, joint_neutral_pose, transform_mat_neutral_pose

    def get_zero_pose_human(self, return_mesh=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        zero_body_pose = torch.zeros((1,(len(smpl_x.joint_part['body'])-1)*3)).float().cuda()
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        shape_param = self.shape_param[None,:]
        face_offset = smpl_x.face_offset[None,:,:].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
        #joint_offset = None 
        output = self.smplx_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        joint_zero_pose = output.joints[0][:smpl_x.joint_num,:] # zero pose human
        if not return_mesh:
            return joint_zero_pose
        else: 
            mesh_zero_pose = output.vertices[0] # zero pose human
            mesh_zero_pose_upsampled = smpl_x.upsample_mesh(mesh_zero_pose) # zero pose human
            return mesh_zero_pose_upsampled, mesh_zero_pose, joint_zero_pose

    def get_transform_mat_joint(self, transform_mat_neutral_pose, joint_zero_pose, smplx_param):
        # 1. 大 pose -> zero pose
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. zero pose -> image pose
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        trans = smplx_param['trans'].view(1,3)

        # forward kinematics
        pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_joint_2 = transform_mat_joint_2[0]
        
        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        transform_mat_joint = torch.bmm(transform_mat_joint_2, transform_mat_joint_1)
        return transform_mat_joint
    
    # def get_transform_mat_joint(self, joint_zero_pose, smplx_param):

    #     # 2. zero pose -> image pose
    #     root_pose = smplx_param['root_pose'].view(1,3)
    #     body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
    #     jaw_pose = smplx_param['jaw_pose'].view(1,3)
    #     leye_pose = smplx_param['leye_pose'].view(1,3)
    #     reye_pose = smplx_param['reye_pose'].view(1,3)
    #     lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
    #     rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
    #     trans = smplx_param['trans'].view(1,3)

    #     # forward kinematics
    #     pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) 
    #     pose = axis_angle_to_matrix(pose)
    #     _, transform_mat_joint= batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smplx_layer.parents)
    #     transform_mat_joint = transform_mat_joint[0]

    #     return transform_mat_joint
    

    def get_transform_mat_vertex(self, transform_mat_joint, nn_vertex_idxs):
        skinning_weight = self.skinning_weight[nn_vertex_idxs,:]
        transform_mat_vertex = torch.matmul(skinning_weight, transform_mat_joint.view(smpl_x.joint_num,16)).view(smpl_x.vertex_num_upsampled,4,4)
        return transform_mat_vertex

    def lbs(self, xyz, transform_mat_vertex, trans):
        xyz = torch.cat((xyz, torch.ones_like(xyz[:,:1])),1) # 大 pose. xyz1
        xyz = torch.bmm(transform_mat_vertex, xyz[:,:,None]).view(smpl_x.vertex_num_upsampled,4)[:,:3]
        #如果想要大字型渲染结果 就把前两句注释掉
        xyz = xyz + trans
        return xyz
    
    def extract_tri_feature(self):
        ## 1. triplane features of all vertices
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_mesh
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / (cfg.triplane_shape_3d[0]/2)
        y = xyz[:,1] / (cfg.triplane_shape_3d[1]/2)
        z = xyz[:,2] / (cfg.triplane_shape_3d[2]/2)
        
        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # smpl_x.vertex_num_upsampled, cfg.triplane_shape[0]*3

        ## 2. triplane features of face vertices
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_mesh[self.is_face,:]
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / (cfg.triplane_face_shape_3d[0]/2)
        y = xyz[:,1] / (cfg.triplane_face_shape_3d[1]/2)
        z = xyz[:,2] / (cfg.triplane_face_shape_3d[2]/2)
        
        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane_face[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane_face[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane_face[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat_face = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # sum(self.is_face), cfg.triplane_shape[0]*3
        
        # combine 1 and 2
        tri_feat[self.is_face] = tri_feat_face
        return tri_feat

    def forward_geo_network(self, tri_feat, smplx_param):
        # poses from smplx parameters
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)

        # combine pose with triplane feature
        pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose))
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose)).view(1,smpl_x.joint_num-1,6).repeat(smpl_x.vertex_num_upsampled,1,1) # without root pose
        pose = pose.view(smpl_x.vertex_num_upsampled, (smpl_x.joint_num-1)*6)
        feat = torch.cat((tri_feat, pose.detach()),1)

        # forward to geometry networks
        geo_offset_feat = self.geo_offset_net(feat)
        mean_offset_offset = self.mean_offset_offset_net(geo_offset_feat) # pose-dependent mean offset of Gaussians
        scale_offset = self.scale_offset_net(geo_offset_feat) # pose-dependent scale of Gaussians
        return mean_offset_offset, scale_offset
    
    def get_mean_offset_offset(self, smplx_param, mean_offset_offset):
        # poses from smplx parameters
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) # without root pose

        # smplx pose-dependent vertex offset
        pose = (axis_angle_to_matrix(pose) - torch.eye(3)[None,:,:].float().cuda()).view(1,(smpl_x.joint_num-1)*9)
        smplx_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(smpl_x.vertex_num_upsampled,3)

        # combine it with regressed mean_offset_offset
        # for face and hands, use smplx offset
        mask = ((self.is_rhand + self.is_lhand + self.is_face_expr) > 0)[:,None].float()
        mean_offset_offset = mean_offset_offset * (1 - mask)
        smplx_pose_offset = smplx_pose_offset * mask
        output = mean_offset_offset + smplx_pose_offset
        return output, mean_offset_offset

    def forward_rgb_network(self, tri_feat, smplx_param, xyz):
        """
        前向传播RGB偏移网络，预测姿态和视角依赖的颜色变化
        
        Args:
            tri_feat: 从三平面提取的特征，包含身份和环境信息
            smplx_param: SMPLX参数字典，包含各种姿态参数
            xyz: 当前姿态下顶点的3D位置（世界坐标系）
        
        Returns:
            rgb_offset: 预测的RGB颜色偏移量，将添加到基础颜色上
        """
        # 1. 从SMPLX参数中提取各种姿态参数并重新整形
        # poses from smplx parameters
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        # 2. 组合所有姿态参数（排除根关节）
        # transform root pose from camera coordinate system to world coordinate system
        pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose))
        # 3. 将轴角表示转换为6D旋转表示（更稳定的优化空间）
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose))
        # 4. 将姿态参数扩展到每个顶点
        pose = pose.view(1,(smpl_x.joint_num-1)*6).repeat(smpl_x.vertex_num_upsampled,1) # smpl_x.vertex_num_upsampled, (smpl_x.joint_num-1)*6
        # 5. 计算世界坐标系中的每顶点法线（用于视角依赖的着色）
        # per-vertex normal in world coordinate system
        with torch.no_grad():  # 不计算梯度，因为法线是几何属性，不应通过颜色网络反向传播
            normal = Meshes(verts=xyz[None], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None]).verts_normals_packed().reshape(smpl_x.vertex_num_upsampled,3) # 创建网格结构来计算法线
            is_cavity = self.is_cavity[:,None].float() # 处理空腔区域：模板网格中空腔区域的法线方向是相反的
            normal = normal * (1 - is_cavity) + (-normal) * is_cavity # cavity has opposite normal direction in the template mesh
        # 6. 准备网络输入特征
        # forward to rgb network 
        feat = torch.cat((tri_feat, pose.detach(), normal.detach()),1) # 拼接三平面特征、姿态参数（分离梯度）和法线（分离梯度）
        # 7. 前向传播到RGB偏移网络
        rgb_offset = self.rgb_offset_net(feat) # pose-dependent rgb offset of Gaussians
        return rgb_offset

    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx

    def forward(self, smplx_param, cam_param=None, is_world_coord=True):
        """
        前向传播函数：从SMPLX参数生成3D高斯资产
        
        Args:
            smplx_param: SMPLX参数字典，包含姿态、表情、平移等信息
            cam_param: 相机参数，用于世界坐标系转换
            is_world_coord: 是否输出世界坐标系下的坐标
        
        Returns:
            assets: 基础高斯资产（位置、尺度、颜色等）
            assets_refined: 精炼后的高斯资产
            offsets: 各种偏移量（用于正则化和分析）
            mesh_neutral_pose: 中性姿态的网格（用于参考）
        """
        # 1. 获取中性姿态的网格和变换矩阵
        # mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _ = self.get_zero_pose_human(return_mesh=True)
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        # mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=False)
        joint_zero_pose = self.get_zero_pose_human()  # 获取零姿态关节位置
        # 2. 从三平面提取特征
        # extract triplane feature
        tri_feat = self.extract_tri_feature()  # 提取三平面编码的特征
        # 3. 获取基础高斯资产
        # get Gaussian assets
        geo_feat = self.geo_net(tri_feat)  # 几何特征
        mean_offset = self.mean_offset_net(geo_feat)  # 高斯均值偏移（相对于中性网格）# mean offset of Gaussians
        scale = self.scale_net(geo_feat) # 高斯尺度（对数空间）# scale of Gaussians
        rgb = self.rgb_net(tri_feat)   # RGB颜色 # rgb of Gaussians
        mean_3d = mesh_neutral_pose + mean_offset # 大 pose  # 最终高斯位置（在大-pose空间）
        # 4. 获取姿态依赖的高斯资产（精炼版本）
        # get pose-dependent Gaussian assets
        mean_offset_offset, scale_offset = self.forward_geo_network(tri_feat, smplx_param)
        scale, scale_refined = torch.exp(scale).repeat(1,3), torch.exp(scale+scale_offset).repeat(1,3) # 尺度处理：从对数空间转换到实际尺度，并复制到xyz三个维度
        mean_combined_offset, mean_offset_offset = self.get_mean_offset_offset(smplx_param, mean_offset_offset)  # 均值偏移处理
        mean_3d_refined = mean_3d + mean_combined_offset # 大 pose
        # 5. 添加SMPLX表情偏移
        # smplx facial expression offset
        smplx_expr_offset = (smplx_param['expr'][None,None,:] * self.expr_dirs).sum(2) # 计算表情引起的顶点偏移：expr参数 × 表情形变基
        mean_3d = mean_3d + smplx_expr_offset # 大 pose # 应用到基础和精炼位置
        mean_3d_refined = mean_3d_refined + smplx_expr_offset # 大 pose
        # 6. 寻找最近邻顶点（用于蒙皮权重分配）
        # get nearest vertex
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        nn_vertex_idxs = knn_points(mean_3d[None,:,:], mesh_neutral_pose_wo_upsample[None,:,:], K=1, return_nn=True).idx[0,:,0] # dimension: smpl_x.vertex_num_upsampled # 使用KNN找到每个高斯点对应的最近网格顶点
        nn_vertex_idxs = self.lr_idx_to_hr_idx(nn_vertex_idxs)  # 将低分辨率索引映射到高分辨率索引
        mask = (self.is_rhand + self.is_lhand + self.is_face) > 0   # 对于手部和面部区域，直接使用原始顶点索引（保持精确的蒙皮权重）
        nn_vertex_idxs[mask] = torch.arange(smpl_x.vertex_num_upsampled).cuda()[mask]
        # 7. 计算变换矩阵并进行线性混合蒙皮（LBS）
        # get transformation matrix of the nearest vertex and perform lbs
        transform_mat_joint = self.get_transform_mat_joint(transform_mat_neutral_pose, joint_zero_pose, smplx_param) # 获取关节变换矩阵
        # transform_mat_vertex = self.get_transform_mat_vertex(transform_mat_neutral_pose, nn_vertex_idxs) 
        transform_mat_vertex = self.get_transform_mat_vertex(transform_mat_joint, nn_vertex_idxs) # 获取顶点变换矩阵（通过蒙皮权重混合关节变换）
        mean_3d = self.lbs(mean_3d, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param # 应用蒙皮变换：将大-pose空间的位置变换到目标姿态空间
        mean_3d_refined = self.lbs(mean_3d_refined, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param
        # 8. 坐标系转换（可选）
        # camera coordinate system -> world coordinate system # 从相机坐标系转换到世界坐标系
        if not is_world_coord: 
            mean_3d = torch.matmul(torch.inverse(cam_param['R']), (mean_3d - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
            mean_3d_refined = torch.matmul(torch.inverse(cam_param['R']), (mean_3d_refined - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        # 9. 前向传播到RGB网络（姿态和视角依赖的颜色）
        # forward to rgb network
        rgb_offset = self.forward_rgb_network(tri_feat, smplx_param, mean_3d_refined)
        rgb, rgb_refined = (torch.tanh(rgb) + 1) / 2, (torch.tanh(rgb + rgb_offset) + 1) / 2 # normalize to [0,1] # 颜色归一化：tanh激活函数将输出映射到[-1,1]，然后转换到[0,1]
        # 10. 组装最终的高斯资产
        # Gaussians and offsets
        rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smpl_x.vertex_num_upsampled,1,1)) # constant rotation # 旋转：使用单位四元数（无旋转）
        opacity = torch.ones((smpl_x.vertex_num_upsampled,1)).float().cuda() # constant opacity # 不透明度：设为全1（完全可见）
        assets = { # 基础资产
                'mean_3d': mean_3d, # 3D位置
                'opacity': opacity,   # 不透明度
                'scale': scale,  # 尺度
                'rotation': rotation,   # 旋转（四元数）
                'rgb': rgb   # 颜色
                }
        assets_refined = { # 精炼资产
                'mean_3d': mean_3d_refined, 
                'opacity': opacity, 
                'scale': scale_refined, 
                'rotation': rotation, 
                'rgb': rgb_refined
                }
        offsets = { # 偏移量（用于正则化损失）
                'mean_offset': mean_offset, # 基础均值偏移
                'mean_offset_offset': mean_offset_offset, # 姿态依赖的均值偏移
                'scale_offset': scale_offset, # 姿态依赖的尺度偏移
                'rgb_offset': rgb_offset # 姿态和视角依赖的颜色偏移
                }
        return assets, assets_refined, offsets, mesh_neutral_pose

class GaussianRenderer(nn.Module):
    """
    高斯渲染器类，基于3D高斯散射(3D Gaussian Splatting)技术
    将3D高斯投影到2D图像平面进行可微分渲染
    """
    def __init__(self):
        super(GaussianRenderer, self).__init__() # 初始化时不包含可学习参数，主要作用是封装渲染流程
    
    def forward(self, gaussian_assets, img_shape, cam_param, bg=None):
        """
        前向渲染函数
        
        Args:
            gaussian_assets: 高斯资产字典，包含3D高斯的所有属性
            img_shape: 输出图像尺寸 (height, width)
            cam_param: 相机参数字典，包含内参和外参
            bg: 背景颜色，默认为None（使用白色）
        
        Returns:
            render_result: 渲染结果字典，包含图像、深度图、掩码等
        """
        # 1. 从高斯资产中提取各个属性
        # assets for the rendering
        mean_3d = gaussian_assets['mean_3d']   # 3D位置 (N, 3)
        opacity = gaussian_assets['opacity']   # 不透明度 (N, 1)
        scale = gaussian_assets['scale']  # 尺度 (N, 3) - XYZ方向
        rotation = gaussian_assets['rotation']  # 旋转 (N, 4) - 四元数表示
        rgb = gaussian_assets['rgb']   # 颜色 (N, 3) - RGB值
        # 2. 创建光栅化器（rasterizer）所需的参数
        # create rasterizer
        # permute view_matrix and proj_matrix following GaussianRasterizer's configuration following below links
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L54
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L55
        fov = get_fov(cam_param['focal'], cam_param['princpt'], img_shape) # 计算视场角(FOV)
        view_matrix = get_view_matrix(cam_param['R'], cam_param['t']).permute(1,0)  # 获取视图矩阵（世界坐标系到相机坐标系）并转置以满足GaussianRasterizer的格式要求
        proj_matrix = get_proj_matrix(cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1,0) # 获取投影矩阵（相机坐标系到裁剪坐标系）并转置
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)  # 计算完整的投影矩阵（视图矩阵 × 投影矩阵）
        cam_pos = view_matrix.inverse()[3,:3] # 计算相机位置（从视图矩阵的逆矩阵中提取）
        if bg is None: # 设置背景颜色（默认为白色）
            bg = torch.ones((3)).float().cuda()
        # 3. 配置光栅化设置
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            bg=bg, 
            scale_modifier=1.0,
            viewmatrix=view_matrix, 
            projmatrix=full_proj_matrix,
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings) # 创建高斯光栅化器实例
        # 4. 准备2D高斯位置（用于梯度跟踪）
        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]  # 高斯点数量
        mean_2d = torch.zeros((point_num,3)).float().cuda()  # 初始化2D位置
        mean_2d.requires_grad = True # 设置梯度跟踪（虽然初始值为0，但渲染过程中会被更新）
        mean_2d.retain_grad()  # 保留梯度以便后续使用
        # 5. 执行光栅化渲染
        # rasterize visible Gaussians to image and obtain their radius (on screen). 
        render_img, radius, render_depthmap, render_mask = rasterizer(
            means3D=mean_3d,
            means2D=mean_2d,
            shs=None,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scale,
            rotations=rotation,
            cov3D_precomp=None)
        # 6. 返回渲染结果
        return {'img': render_img, # 渲染的RGB图像 (3, H, W)
                'depthmap': render_depthmap, # 深度图 
                'mask': render_mask, # 渲染掩码（alpha通道）
                'mean_2d': mean_2d, # 2D投影位置（用于梯度传播）
                'is_vis': radius > 0, # 可见性掩码（哪些高斯点对渲染有贡献）
                'radius': radius}  # 在屏幕上的半径（用于重要性排序）

class SMPLXParamDict(nn.Module):
    """
    SMPLX参数字典类，用于管理和优化SMPLX模型的参数
    支持从预定义参数初始化，并提供可优化的参数接口
    """
    def __init__(self):
        super(SMPLXParamDict, self).__init__() # 初始化时不包含参数，需要通过init方法加载

    # initialize SMPL-X parameters of all frames  # 初始化所有帧的SMPLX参数
    # used to train models from scratch # 用于从头开始训练模型
    def init(self, smplx_params): # 初始化SMPLX参数，将其转换为可优化的ParameterDict
        _smplx_params = {} # 临时字典用于构建参数结构
        for capture_id in smplx_params.keys():  # 遍历所有捕获会话(capture)
            _smplx_params[capture_id] = nn.ParameterDict({}) # 为每个capture创建ParameterDict
            for frame_idx in smplx_params[capture_id].keys():  # 遍历该capture中的所有帧
                _smplx_params[capture_id][str(frame_idx)] = nn.ParameterDict({}) # 为每帧创建ParameterDict
                for param_name in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']: # 遍历所有SMPLX参数类型
                    if 'pose' in param_name:  # 姿态参数：转换为6D旋转表示（更稳定的优化空间）
                        _smplx_params[capture_id][str(frame_idx)][param_name] = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(smplx_params[capture_id][frame_idx][param_name].cuda())))
                    else: # 非姿态参数（表情、平移）：直接使用
                        _smplx_params[capture_id][str(frame_idx)][param_name] = nn.Parameter(smplx_params[capture_id][frame_idx][param_name].cuda())
        self.smplx_params = nn.ParameterDict(_smplx_params) # 将临时字典转换为nn.ParameterDict并注册为模块参数

    def get_optimizable_params(self):
        """
        获取可优化的参数列表，用于配置优化器
        
        Returns:
            optimizable_params: 包含所有可优化参数的列表，每个参数都有独立的学习率配置
        """
        optimizable_params = []
        for capture_id in self.smplx_params.keys():  # 遍历所有参数层级：capture → frame → parameter
            for frame_idx in self.smplx_params[capture_id].keys():
                for param_name in self.smplx_params[capture_id][frame_idx].keys():
                    optimizable_params.append({'params': [self.smplx_params[capture_id][frame_idx][param_name]], 'name': 'smplx_' + param_name + '_' + capture_id + '_' + frame_idx, 'lr': cfg.smplx_param_lr}) # 为每个参数创建优化配置
        return optimizable_params

    def forward(self, capture_ids, frame_idxs):
        """
        前向传播，根据给定的capture_id和frame_idx获取对应的SMPLX参数
        
        Args:
            capture_ids: 捕获会话ID列表
            frame_idxs: 帧索引列表
            
        Returns:
            out: SMPLX参数字典列表，每个元素对应一个请求的帧
        """
        out = []
        for capture_id, frame_idx in zip(capture_ids, frame_idxs): # 遍历所有请求的(capture_id, frame_idx)对
            capture_id = str(capture_id) # 确保字符串类型
            frame_idx = str(int(frame_idx)) # 确保字符串类型（先转int再转str避免浮点数问题）
            smplx_param = {}  # 当前帧的参数字典
            for param_name in self.smplx_params[capture_id][frame_idx].keys():  # 遍历该帧的所有参数
                if 'pose' in param_name:  # 姿态参数：从6D表示转换回轴角表示（SMPLX模型需要的格式）
                    smplx_param[param_name] = matrix_to_axis_angle(rotation_6d_to_matrix(self.smplx_params[capture_id][frame_idx][param_name]))
                else: # 非姿态参数：直接使用
                    smplx_param[param_name] = self.smplx_params[capture_id][frame_idx][param_name]
            out.append(smplx_param) # 添加到输出列表
        return out
