python setup.py build develop
python -m torch.distributed.launch --nproc_per_node=8 --master_port 47779  \
       tore/tools/run_tore_metro_bodymesh.py \
       --run_eval_only \
       --val_yaml /code/posemae/MeshGraphormer_data/datasets/human3.6m/valid.protocol2.yaml \
       --num_workers 4 \
       --per_gpu_eval_batch_size 32 \
       --arch hrnet-w64 \
       --output_dir eval/metro_h64_gtr_h36m/ \
       --resume_checkpoint checkpoints/metro_h64_gtr_37.1.bin \
       --num_hidden_layers 4 \
       --num_attention_heads 4 \
       --input_feat_dim 2051,512,128 \
       --hidden_feat_dim 1024,256,128

