# TORE: Token Reduction for Efficient Human Mesh Recovery with Transformer

[[Project Page](https://frank-zy-dou.github.io/projects/Tore/index.html)][[Paper](https://arxiv.org/abs/2211.10705)][[Code](https://github.com/Frank-ZY-Dou/TORE)]


This is the official PyTorch implementation of [TORE: Token Reduction for Efficient Human Mesh Recovery with Transformer](https://arxiv.org/abs/2211.10705) (ICCV 2023). 


## Installation
Follow FASTMETRO installation (CUDA 10.1). CUDA 11.1 is not currently supported, due to OpenDR not supporting it.

We recommend create a new conda environment for this project.

```
# Create a conda environment, activate the environment and install PyTorch via conda
conda create --name tore python=3.8
conda activate tore
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

# Install OpenDR
pip install git+https://gitlab.eecs.umich.edu/ngv-python-modules/opendr.git

# Install FastMETRO
git clone --recursive https://github.com/Frank-ZY-Dou/TORE.git
cd TORE
python setup.py build develop

# Install requirements
pip install -r requirements.txt

# Install manopth
pip install ./manopth/.
```

We also provide a docker container for easy environment installation.



Beyond setting up the environment, our repository needs additional files to work. 

Please download the models folder from this [link](https://drive.google.com/drive/folders/1yhgKYUy0OxZ5kiC4iiiEWLPYGLwdXLAT?usp=drive_link) and set up `TORE/models` according to the following file tree:

```
models
├── efficientnet
│   └── efficientnet-b0-355c32eb.pth
├── hrnet
│   ├── cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
│   └── hrnetv2_w64_imagenet_pretrained.pth
└── resnet
    └── resnet50-0676ba61.pth
```

Please also download the data folder from this [link](https://drive.google.com/drive/folders/1oOUqxSMl3vTlObWr7vr8SZQ2Nva-ywAG?usp=drive_link) and set up `TORE/tore/modeling/data` according to the following file tree:

```
tore/modeling/data
├── J_regressor_extra.npy
├── J_regressor_h36m_correct.npy
├── MANO_RIGHT.pkl
├── README.md
├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
├── config.py
├── mano_195_adjmat_indices.pt
├── mano_195_adjmat_size.pt
├── mano_195_adjmat_values.pt
├── mano_downsampling.npz
├── mesh_downsampling.npz
├── smpl_431_adjmat_indices.pt
├── smpl_431_adjmat_size.pt
├── smpl_431_adjmat_values.pt
└── smpl_431_faces.npy
```

## Checkpoints

We provide various pre-trained checkpoints for inference and fine-tuning.

### Human3.6M

| Name                                      | PA-MPJPE | GFLOPs | Link                                                         |
| ----------------------------------------- | -------- | ------ | ------------------------------------------------------------ |
| FastMETRO + HRNet-w64 + TORE (@20%)       | 36.4     | 30.2   | [Google Drive](https://drive.google.com/file/d/1mly-hkm3oW_UbGL20cQ-MVJvtpigXAhr/view?usp=drive_link) |
| FastMETRO + ResNet50 + TORE (@20%)        | 40.5     | 5.4    | [Google Drive](https://drive.google.com/file/d/1kPPAESrfl7sI0NvMnXDkwFec6ATqWMrX/view?usp=drive_link) |
| FastMETRO + EfficientNet-b0 + TORE (@20%) | 43.9     | 1.7    | [Google Drive](https://drive.google.com/file/d/1RrXOSRDqWCDzm0s8LDLUZrXvSrzOVQA4/view?usp=drive_link) |
| METRO + HRNet-w64 + TORE                  | 37.1     | 30.2   | [Google Drive](https://drive.google.com/file/d/1fd9isJin_zi_rxNV3okT5B1puw-AfTUr/view?usp=drive_link) |
| METRO + ResNet50 + TORE                   | 42.0     | 5.4    | [Google Drive](https://drive.google.com/file/d/10gQtqWJVdmWmOjMkOTykinLQGTu96sF8/view?usp=drive_link) |

### 3DPW

| Name                                | PA-MPJPE | GFLOPs | Link                                                         |
| ----------------------------------- | -------- | ------ | ------------------------------------------------------------ |
| FastMETRO + HRNet-w64 + TORE (@20%) | 44.4     | 30.2   | [Google Drive](https://drive.google.com/file/d/17icIUL7FUdCMl6cKJNApwhTvrjNBClPN/view?usp=drive_link) |

## Inference
Use the following shell command for inference.

```
python ./tore/tools/tore_inference_fm.py \
       --resume_checkpoint [your_checkpoint.bin] \
       --image_file_or_path [image folder or image file]
```

 A template is provided in `inference_tore_fm.sh`.  We recommend using the `FastMETRO + HRNet-w64 + TORE (@20%) ` checkpoint, due to its strong generalizing ability on in-the-wild images.


## Experiments
To train the TORE model, we need to download additional datasets. Please follow Part 5 in [DOWNLOAD.md](https://github.com/microsoft/MeshTransformer/blob/main/docs/DOWNLOAD.md) of METRO to download the datasets.

Then, use the following shell command for training TORE with FastMETRO:

```
python setup.py build develop
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=44444 \
       tore/tools/run_tore_fm_bodymesh.py \
       --train_yaml your_dataset_folder/Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
       --val_yaml your_dataset_folder/human3.6m/valid.protocol2.yaml \
       --num_workers 4 \
       --per_gpu_train_batch_size 16 \
       --per_gpu_eval_batch_size 16 \
       --lr 1e-4 \
       --arch efficientnet-b0 \
       --num_train_epochs 60 \
       --output_dir your_output_folder \
       --keep_ratio 0.8 \
       --model_name 'FastMETRO_L' \
       --itp_loss_weight 1e-3 \
       --edge_and_normal_vector_loss "false"
```

An example is provided in `train_tore_fm.sh`.

Use the following shell command for training TORE with METRO:

```
python setup.py build develop
python -m torch.distributed.launch --nproc_per_node=8 \
       tore/tools/run_tore_m_bodymesh.py \
       --train_yaml your_dataset_folder/Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
       --val_yaml your_dataset_folder/human3.6m/valid.protocol2.yaml \
       --arch resnet50 \
       --num_workers 4 \
       --per_gpu_train_batch_size 32 \
       --per_gpu_eval_batch_size 32 \
       --num_hidden_layers 4 \
       --num_attention_heads 4 \
       --lr 1e-4 \
       --num_train_epochs 200 \
       --input_feat_dim 2051,512,128 \
       --hidden_feat_dim 1024,256,128 \
       --output_dir your_output_folder
```

Use `--arch=hrnet-w64` for HRNet-W64 backbone, `--arch=resnet50`  for ResNet50 backbone, and `--arch=efficientnet-b0`  for EfficientNet-b0 backbone.


## Contributing 

Please note that enhancing mesh quality can be achieved by applying a [SMPL parameter regressor](https://github.com/postech-ami/FastMETRO/blob/main/src/modeling/model/smpl_param_regressor.py).

We welcome contributions and suggestions. 


## Citations
If you find our work useful in your research, please consider citing:

```bibtex
@InProceedings{Dou_2023_ICCV,
    author    = {Dou, Zhiyang and Wu, Qingxuan and Lin, Cheng and Cao, Zeyu and Wu, Qiangqiang and Wan, Weilin and Komura, Taku and Wang, Wenping},
    title     = {TORE: Token Reduction for Efficient Human Mesh Recovery with Transformer},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {15143-15155}
}
```


## License

Our research code is released under the MIT license. 

We use submodules from third parties, such as [huggingface/transformers](https://github.com/huggingface/transformers) and [hassony2/manopth](https://github.com/hassony2/manopth). Please see [NOTICE](NOTICE.md) for details. 

We note that any use of SMPL models and MANO models are subject to **Software Copyright License for non-commercial scientific research purposes**. See [SMPL-Model License](https://smpl.is.tue.mpg.de/modellicense) and [MANO License](https://mano.is.tue.mpg.de/license) for details.



## Acknowledgments

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[huggingface/transformers](https://github.com/huggingface/transformers) 

[HRNet/HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification) 

[nkolot/GraphCMR](https://github.com/nkolot/GraphCMR) 

[akanazawa/hmr](https://github.com/akanazawa/hmr) 

[MandyMo/pytorch_HMR](https://github.com/MandyMo/pytorch_HMR) 

[hassony2/manopth](https://github.com/hassony2/manopth) 

[hongsukchoi/Pose2Mesh_RELEASE](https://github.com/hongsukchoi/Pose2Mesh_RELEASE) 

[mks0601/I2L-MeshNet_RELEASE](https://github.com/mks0601/I2L-MeshNet_RELEASE) 

[open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) 

[microsoft/MeshTransformer](https://github.com/microsoft/MeshTransformer)

[postech-ami/FastMETRO](https://github.com/postech-ami/FastMETRO)

[lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)