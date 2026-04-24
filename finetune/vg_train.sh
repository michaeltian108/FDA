gpu=0,1,2,3
IFS=',' read -r -a array <<< "$gpu"
num_nodes=${#array[@]}


OMP_NUM_THREADS=${num_nodes} CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.run --nproc_per_node=${num_nodes}  --master_port ${PORT} Grounding_fda.py \
    --config ./configs/Grounding.yaml \
    --dif FDA --lbd 1 --vocab short \
    --output_dir OUTPUT_PATH \
    --gradcam_mode itm --block_num 8 \
    --checkpoint PATH_TO_CHECKPOINT/ALBEF_14M.pth 
