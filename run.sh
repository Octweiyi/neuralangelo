EXPERIMENT=$1
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${EXPERIMENT} \
    --config=${CONFIG} \
    --wandb \
    --wandb_name ${EXPERIMENT} \
    --show_pbar
