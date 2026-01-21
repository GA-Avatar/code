## Install
* To install a conda environment and necessary packages, run below.
```
conda env create -f environment.yml
conda activate gaavatar
```

* Also, to download and install third-party modules.
```
bash install.sh
```


# Creating an avatar from [X-Humans dataset](https://skype-line.github.io/projects/X-Avatar/)

* This branch includes **avatar creation pipeline and animation function**.
* This code includes **avatar creation pipeline and animation function** only when exact foreground mask is given like [X-Humans dataset](https://skype-line.github.io/projects/X-Avatar/).


## Directory
```
${ROOT}
|-- main
|-- common
|-- |-- utils/human_model_files
|-- |-- |-- smplx/SMPLX_FEMALE.npz
|-- |-- |-- smplx/SMPLX_MALE.npz
|-- |-- |-- smplx/SMPLX_NEUTRAL.npz
|-- |-- |-- smplx/MANO_SMPLX_vertex_ids.pkl
|-- |-- |-- smplx/SMPL-X__FLAME_vertex_ids.npy
|-- |-- |-- smplx/smplx_flip_correspondences.npz
|-- |-- |-- flame/flame_dynamic_embedding.npy
|-- |-- |-- flame/FLAME_FEMALE.pkl
|-- |-- |-- flame/FLAME_MALE.pkl
|-- |-- |-- flame/FLAME_NEUTRAL.pkl
|-- |-- |-- flame/flame_static_embedding.pkl
|-- |-- |-- flame/FLAME_texture.npz
|-- data
|-- |-- XHumans
|-- |-- |-- data/00028
|-- |-- |-- data/00034
|-- |-- |-- data/00087
|-- tools
|-- output
```
* `main` contains high-level code for the avatar creation/animation and configurations.
* `common` contains kernel code. Download SMPL-X 1.1 version from [here](https://smpl-x.is.tue.mpg.de/download.php). Download FLAME 2020 version from [here](https://flame.is.tue.mpg.de/download.php).
* `data` contains data loading code.
* `tools` contains pre-processing and evaluation code.
* `output` contains log, visualized outputs, and fitting result.

## XHumans videos
* You can download XHumans data from [here](https://drive.google.com/drive/folders/1TalHPkbohPoTPNawVi2gbj6M8nAyYAE9?usp=sharing).
* Go to `preprocess/tools` folder. This step will generate geometric supervision information (depth information and normal information).
* Run `export CUDA_VISIBLE_DEVICES=$GPU_ID` where `$GPU_ID` is your desired GPU index. 
* Run `python generated.py --root_path $ROOT_PATH`, where `$ROOT_PATH` is an **absolute path** to the subject. In the case of above directory, `$ROOT_PATH` is `$ROOT/data/subjects/$SUBJECT_NAME`.

## Train
* Set `dataset='XHumans'` in `main/config.py`.
* Go to `main` folder and run `python train.py --subject_id $SUBJECT_ID`. The checkpoints are saved in `output/model/$SUBJECT_ID`.
* You can use one of `00028`, `00034`, and `00087` for `$SUBJECT_ID`.

## Test and evaluation
* Set `dataset='XHumans'` in `main/config.py`.
* You can see test results on the testing frames by running `python test.py --subject_id $SUBJECT_ID --test_epoch 24`. The results are saved to `output/result/$SUBJECT_ID`.
* For the evaluation of the X-Humans dataset, go to `tools` folder and run `python eval_xhumans.py --output_path ../output/result/$SUBJECT_ID --subject_id $SUBJECT_ID`.


