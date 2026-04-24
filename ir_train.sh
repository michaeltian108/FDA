data=flickr
gpu=0,1,2,3

IFS=',' read -r -a array <<< "$gpu"
num_nodes=${#array[@]}

CUDA_VISIBLE_DEVICES=$gpu  OMP_NUM_THREADS=${num_nodes} python -m torch.distributed.run --nproc_per_node=${num_nodes}  Retrieval_fda.py \
    --config ./configs/Retrieval_${data}.yaml \
    --lbd 1 --dif FDA --vocab short \
    --output_dir OUTPUT_DIR \
    --checkpoint PATH_TO_CHECKPOINT/ALBEF_14M.pth 

