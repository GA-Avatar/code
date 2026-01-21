import torch
import argparse
from tqdm import tqdm
import numpy as np
from config import cfg
from base import Tester
import os
import os.path as osp
import cv2
from utils.smpl_x import smpl_x
from pytorch3d.io import save_obj

def save_color_depthmap_transparent(depthmap, save_path, colormap=cv2.COLORMAP_INFERNO):
    """
    保存彩色透明背景的深度图
    
    Args:
        depthmap: 深度图数据 (2D numpy array)
        save_path: 保存路径
        colormap: OpenCV色彩映射，默认为INFERNO
    """
    # 确保深度图是2D的
    depthmap_squeezed = depthmap.squeeze()
    
    # 创建mask（深度值大于0的区域）
    mask = depthmap_squeezed > 0
    
    if np.any(mask):
        # 提取前景深度值
        depth_foreground = depthmap_squeezed[mask]
        
        # 归一化深度值
        min_val = depth_foreground.min()
        max_val = depth_foreground.max()
        
        # 归一化到 [0, 1] 范围，并反转（深度值大的显示为"热"色）
        depth_normalized = 1 - ((depth_foreground - min_val) / (max_val - min_val + 1e-8))
        
        # 转换为 0-255 范围
        depth_normalized_255 = (depth_normalized * 255).astype(np.uint8)
        
        # 应用色彩映射
        depth_colored = cv2.applyColorMap(depth_normalized_255, colormap)
        
        # 创建RGBA图像（带透明通道）
        height, width = depthmap_squeezed.shape
        depth_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        
        # 设置RGB颜色
        depth_colored_reshaped = depth_colored.reshape(-1, 3)
        depth_rgba[mask, :3] = depth_colored_reshaped
        
        # 设置Alpha通道：前景=255（不透明），背景=0（完全透明）
        depth_rgba[mask, 3] = 255
        
        # 保存透明背景的彩色深度图
        cv2.imwrite(save_path, depth_rgba)
        
    else:
        # 如果没有深度数据，保存完全透明的图像
        height, width = depthmap_squeezed.shape
        depth_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        cv2.imwrite(save_path, depth_rgba)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--output_dir', type=str, default='./outputs',help='Directory to save results and logs')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id, False, args.output_dir)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
 
    for itr, data in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            out = tester.model(data, 'test', epoch=None)
            # out = tester.model(data, 'test')
        
        # save
        human_depthmap_refined =  out['human_depthmap_refined'].cpu().numpy()
        human_normalmap_refined = out['human_normalmap_refined'].cpu().numpy()
        human_img = out['human_img'].cpu().numpy()
        human_img_refined = out['human_img_refined'].cpu().numpy()
        human_mask_refined = out['human_mask_refined'].cpu().numpy()
        human_face_img = out['human_face_img'].cpu().numpy()
        human_face_img_refined = out['human_face_img_refined'].cpu().numpy()
        smplx_mesh = out['smplx_mesh'].cpu().numpy()
        batch_size = human_img.shape[0]
        for i in range(batch_size):
            capture_id = str(data['capture_id'][i])
            frame_idx = int(data['frame_idx'][i])
            save_root_path = osp.join(cfg.result_dir, 'test', capture_id)
            os.makedirs(save_root_path, exist_ok=True)

            save_color_depthmap_transparent(human_depthmap_refined[i], osp.join(save_root_path, f"{frame_idx}_human_depthmap.png"))
            cv2.imwrite(osp.join(save_root_path, f"{frame_idx}_human_normalmap.png"),np.dstack([(human_normalmap_refined[i].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8),(human_mask_refined[i] * 255).astype(np.uint8)]))# cv2.imwrite(osp.join(save_root_path, f"{frame_idx}_human_depthmap_refined.png"), ((human_depthmap_refined[i].squeeze() > 0) * (255 - ((human_depthmap_refined[i].squeeze() - human_depthmap_refined[i][human_depthmap_refined[i] > 0].min()) / (human_depthmap_refined[i][human_depthmap_refined[i] > 0].max() - human_depthmap_refined[i][human_depthmap_refined[i] > 0].min() + 1e-8) * 255))).astype(np.uint8))
            # cv2.imwrite(osp.join(save_root_path, f"{frame_idx}_human_normalmap_refined_whitebg.png"),(human_mask_refined[i][...,None] * human_normalmap_refined[i].transpose(1,2,0)[:,:,::-1] + (1 - human_mask_refined[i][...,None])) * 255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_normalmap_black.png'), human_normalmap_refined[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human.png'), human_img[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_refined.png'), human_img_refined[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_gt.png'), data['img'][i].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_face.png'), human_face_img[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_face_refined.png'), human_face_img_refined[i].transpose(1,2,0)[:,:,::-1]*255)

            save_obj(osp.join(save_root_path, str(frame_idx) + '_smplx.obj'), torch.FloatTensor(smplx_mesh[i]), torch.LongTensor(smpl_x.face))

    
if __name__ == "__main__":
    main()


