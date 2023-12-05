python setup.py build develop
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port 47779  \
       tore/tools/run_tore_fastmetro_bodymesh.py \
       --run_eval_only \
       --val_yaml /code/posemae/MeshGraphormer_data/datasets/human3.6m/valid.protocol2.yaml \
       --num_workers 4 \
       --per_gpu_eval_batch_size 16 \
       --arch hrnet-w64 \
       --output_dir eval/h64_gtr_itp08_h36m/ \
       --keep_ratio 0.8 \
       --resume_checkpoint checkpoints/h64_gtr_itp0.8_36.4.bin
