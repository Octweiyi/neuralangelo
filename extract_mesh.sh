obj_name=$1
root_path=$2
CHECKPOINT=logs/${obj_name}/epoch_15625_iteration_000500000_checkpoint.pt
OUTPUT_MESH=${root_path}/${obj_name}.ply
CONFIG=logs/${obj_name}/config.yaml
RESOLUTION=256
BLOCK_RES=256
GPUS=1  # use >1 for multi-GPU mesh extraction
torchrun --nproc_per_node=${GPUS} projects/neuralangelo/scripts/extract_mesh.py \
    --config=${CONFIG} \
    --checkpoint=${CHECKPOINT} \
    --output_file=${OUTPUT_MESH} \
    --resolution=${RESOLUTION} \
    --block_res=${BLOCK_RES}

# to save cuda memory the extract mesh process is devided into sevaral blocks to do it.
# the RESOLUTION is marching cube resolution.
# the BLOCK_RES is cubes number in each block.
