import os
import pickle
import torch
import numpy as np

# 文件夹路径
smpl_dir = '/work/workspace/data_l/ExAvatar_RELEASE-X-Humans/data/XHumans/data/00028/train/Take2/SMPL'

# 获取所有 mesh-fXXXXX_smpl.pkl 文件，并按帧排序
smpl_files = sorted([f for f in os.listdir(smpl_dir) if f.endswith('_smpl.pkl')])

body_pose_list = []
trans_list = []
beta_list = []

for f in smpl_files:
    path = os.path.join(smpl_dir, f)
    with open(path, 'rb') as fp:
        smpl_data = pickle.load(fp)
    
    # 拼接 global_orient + body_pose -> 72
    pose72 = np.concatenate([smpl_data['global_orient'], smpl_data['body_pose']], axis=0)
    body_pose_list.append(pose72.astype(np.float32))
    
    # transl
    trans_list.append(smpl_data['transl'].astype(np.float32))
    
    # beta
    beta_list.append(smpl_data['betas'].astype(np.float32))

# 转 torch tensor
body_pose = torch.from_numpy(np.stack(body_pose_list, axis=0))  # (帧数, 72)
trans = torch.from_numpy(np.stack(trans_list, axis=0))          # (帧数, 3)
beta = torch.from_numpy(np.stack(beta_list, axis=0))            # (帧数, 10)

# 保存为 pth 文件
output_path = '/work/workspace/data_l/ExAvatar_RELEASE-X-Humans/data/XHumans/data/00028/train/Take2/smpl_parms.pth'
torch.save({
    'body_pose': body_pose,
    'beta': beta,
    'trans': trans
}, output_path)

print(f'✅ 保存完成: {output_path}')
print(f'body_pose: {body_pose.shape}, beta: {beta.shape}, trans: {trans.shape}')
