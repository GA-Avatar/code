import os
import os.path as osp
from glob import glob
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path
cur_path = osp.dirname(osp.abspath(__file__))
if root_path[-1] == '/':
    subject_id = root_path.split('/')[-2]
else:
    subject_id = root_path.split('/')[-1]
    
    # sapiens (depth using binary masks)
    os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
    cmd = 'python run_sapiens_depth.py --root_path ' + split_root_path + ' --vis_format video'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running sapiens depth (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)

    # sapiens (normal using binary masks)
    os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
    cmd = 'python run_sapiens_normal.py --root_path ' + split_root_path + ' --vis_format video'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running sapiens normal (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)



