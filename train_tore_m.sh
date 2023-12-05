python setup.py build develop
python -m torch.distributed.launch --nproc_per_node=8 \
       tore/tools/run_tore_m_bodymesh.py \
       --train_yaml /code/posemae/MeshGraphormer_data/datasets/Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
       --val_yaml /code/posemae/MeshGraphormer_data/datasets/human3.6m/valid.protocol2.yaml \
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
       --output_dir output/metro_r50_gtr_test/