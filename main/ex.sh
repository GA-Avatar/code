#!/bin/bash

# 定义要执行的命令
COMMAND_ex1="python test.py --subject_id 00028 --test_epoch 24 --output_dir /work/workspace/data_l/ExAvatar_RELEASE-X-Humans/00028_depth_0.1"
COMMAND_ex2="python test.py --subject_id 00034 --test_epoch 24 --output_dir /work/workspace/data_l/ExAvatar_RELEASE-X-Humans/00034_depth_0.1"
COMMAND_ex3="python test.py --subject_id 00087 --test_epoch 24 --output_dir /work/workspace/data_l/ExAvatar_RELEASE-X-Humans/00087_depth_0.1"
COMMAND_ex4="python test.py --subject_id 00028 --test_epoch 24 --output_dir /work/workspace/data_l/ExAvatar_RELEASE-X-Humans/00028"
COMMAND_ex5="python test.py --subject_id 00034 --test_epoch 24 --output_dir /work/workspace/data_l/ExAvatar_RELEASE-X-Humans/00034"
COMMAND_ex6="python test.py --subject_id 00087 --test_epoch 24 --output_dir /work/workspace/data_l/ExAvatar_RELEASE-X-Humans/00087"
# 循环执行命令五次
$COMMAND_ex1
$COMMAND_ex2
$COMMAND_ex3
$COMMAND_ex4
$COMMAND_ex5
$COMMAND_ex6

