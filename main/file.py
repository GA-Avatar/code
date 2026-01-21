import os
import shutil
import pickle
import torch
import numpy as np

# ----------------------------
# 用户配置
# ----------------------------
dataset_root = '/work/workspace/data_l/ExAvatar_RELEASE-X-Humans/data/XHumans/data/00087/train'  # 大数据集根目录
output_image_dir = '/work/workspace/data_l/ExAvatar_RELEASE-X-Humans/data/XHumans/data/00087gs/train/images'
output_mask_dir = '/work/workspace/data_l/ExAvatar_RELEASE-X-Humans/data/XHumans/data/00087gs/train/masks'
output_smpl_path = '/work/workspace/data_l/ExAvatar_RELEASE-X-Humans/data/XHumans/data/00087gs/train/smpl_parms.pth'
output_cameras_path = '/work/workspace/data_l/ExAvatar_RELEASE-X-Humans/data/XHumans/data/00087gs/train/cam_parms.npz'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# ----------------------------
# 遍历子文件夹
# ----------------------------
body_pose_list = []
trans_list = []
beta = None
image_counter = 0

all_extrinsics = []
all_intrinsics = []

subfolders = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])

for sub in subfolders:
    sub_path = os.path.join(dataset_root, sub)
    render_dir = os.path.join(sub_path, 'render')
    smpl_dir = os.path.join(sub_path, 'SMPL')
    
    image_dir = os.path.join(render_dir, 'image')
    mask_dir = os.path.join(render_dir, 'masks')
    
    # 获取图片和mask文件名，并排序
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    smpl_files = sorted([f for f in os.listdir(smpl_dir) if f.endswith('_smpl.pkl')])
    
    # 检查数量一致
    assert len(image_files) == len(mask_files) == len(smpl_files), f"{sub} 中数量不匹配"
    
    # ---------------- 读取 cameras.npz ----------------
    cameras_path = os.path.join(render_dir, 'cameras.npz')
    cam_data = np.load(cameras_path)
    
    # 假设 cameras.npz 里有 'extrinsics' 和 'intrinsics'
    # extrinsics: (N_frames, 4,4)
    # intrinsics: (3,3) 或 (N_frames,3,3) 如果已经是帧数个
    extrinsics = cam_data['extrinsic']  # shape (num_frames, 4,4)
    intrinsics = cam_data['intrinsic']  # shape (3,3) 或 (num_frames,3,3)
    
    # 如果 intrinsics 只有一个，复制到每帧
    if intrinsics.ndim == 2:  # (3,3)
        intrinsics = np.repeat(intrinsics[None], len(image_files), axis=0)  # (N_frames,3,3)
    
    assert extrinsics.shape[0] == len(image_files), f"{sub} extrinsics帧数与图片不匹配"
    assert intrinsics.shape[0] == len(image_files), f"{sub} intrinsics帧数与图片不匹配"
    
    all_extrinsics.append(extrinsics.astype(np.float32))
    all_intrinsics.append(intrinsics.astype(np.float32))
    
    # ---------------- 遍历每帧 ----------------
    for i in range(len(image_files)):
        # ---------------- copy image ----------------
        src_image_path = os.path.join(image_dir, image_files[i])
        dst_image_path = os.path.join(output_image_dir, f'{image_counter:06d}.png')
        shutil.copy(src_image_path, dst_image_path)
        
        # ---------------- copy mask ----------------
        src_mask_path = os.path.join(mask_dir, mask_files[i])
        dst_mask_path = os.path.join(output_mask_dir, f'{image_counter:06d}.png')
        shutil.copy(src_mask_path, dst_mask_path)
        
        # ---------------- load smpl ----------------
        smpl_path = os.path.join(smpl_dir, smpl_files[i])
        with open(smpl_path, 'rb') as fp:
            smpl_data = pickle.load(fp)
        
        # 拼接 global_orient + body_pose -> 72
        pose72 = np.concatenate([smpl_data['global_orient'], smpl_data['body_pose']], axis=0)
        body_pose_list.append(pose72.astype(np.float32))
        trans_list.append(smpl_data['transl'].astype(np.float32))
        
        # beta 只取第一帧
        if beta is None:
            beta = torch.from_numpy(smpl_data['betas'].astype(np.float32)).unsqueeze(0)
        
        image_counter += 1

# ----------------------------
# 保存 SMPL pth 文件
# ----------------------------
body_pose = torch.from_numpy(np.stack(body_pose_list, axis=0))
trans = torch.from_numpy(np.stack(trans_list, axis=0))

torch.save({
    'body_pose': body_pose,
    'beta': beta,
    'trans': trans
}, output_smpl_path)

# ----------------------------
# 保存 cameras npz 文件
# ----------------------------
all_extrinsics = np.concatenate(all_extrinsics, axis=0)  # (总帧数,4,4)
all_intrinsics = np.concatenate(all_intrinsics, axis=0)  # (总帧数,3,3)

np.savez(output_cameras_path, extrinsic=all_extrinsics, intrinsic=all_intrinsics)

print(f'✅ 合并完成！')
print(f'图片数量: {image_counter}, body_pose: {body_pose.shape}, beta: {beta.shape}, trans: {trans.shape}')
print(f'camera extrinsics: {all_extrinsics.shape}, intrinsics: {all_intrinsics.shape}')
print(f'图片路径: {output_image_dir}, mask路径: {output_mask_dir}, SMPL路径: {output_smpl_path}, cameras路径: {output_cameras_path}')

