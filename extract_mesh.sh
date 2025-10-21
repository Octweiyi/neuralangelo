CHECKPOINT=logs/assemble_with_base/epoch_08000_iteration_000360000_checkpoint.pt
OUTPUT_MESH=dataset/res/assemble_10w_iter.ply
CONFIG=logs/assemble/config.yaml
RESOLUTION=2048
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
