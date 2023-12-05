python setup.py build develop
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=44444 \
       tore/tools/run_tore_fm_bodymesh.py \
       --train_yaml /code/posemae/MeshGraphormer_data/datasets/Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
       --val_yaml /code/posemae/MeshGraphormer_data/datasets/human3.6m/valid.protocol2.yaml \
       --num_workers 4 \
       --per_gpu_train_batch_size 16 \
       --per_gpu_eval_batch_size 16 \
       --lr 1e-4 \
       --arch efficientnet-b0 \
       --num_train_epochs 60 \
       --output_dir output/output_eb0_gtr_itp0.8_test/ \
       --keep_ratio 0.8 \
       --model_name 'FastMETRO_L' \
       --itp_loss_weight 1e-3 \
       --edge_and_normal_vector_loss "false"
