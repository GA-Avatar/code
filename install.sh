#!/bin/bash
set -e

# Move into third_modules directory
pushd third_modules

# ===== sapiens =====
echo "Downloading sapiens..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/sapiens.zip
unzip -o sapiens.zip
rm -f sapiens.zip
pushd sapiens/lite/scripts/demo/torchscript/checkpoints
wget https://huggingface.co/facebook/sapiens-pose-bbox-detector/resolve/c844c2df76f1613d7c5e2910d8bf30039a55a386/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth # human detection
wget https://huggingface.co/facebook/sapiens-depth-1b-torchscript/resolve/main/sapiens_1b_render_people_epoch_88_torchscript.pt2 # depth
wget https://huggingface.co/facebook/sapiens-normal-1b-torchscript/resolve/main/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2 # normal
wget https://huggingface.co/facebook/sapiens-seg-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 # seg
FILENAME="sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2" # pose
huggingface-cli download noahcao/sapiens-pose-coco \
  --include "sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/${FILENAME}" \
  --repo-type model \
  --local-dir ./tmp_hf_download \
  --local-dir-use-symlinks False \
  --resume-download
mv "./tmp_hf_download/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/${FILENAME}" "./${FILENAME}"
rm -rf ./tmp_hf_download
popd  # Return to third_modules
