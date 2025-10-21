# -----------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -----------------------------------------------------------------------------

# usage: preprocess.sh <sequence_name> <full_video_path> <downsample_rate> <scene_type>

data_path=$2
scene_type=object
if [ -d "${data_path}/sparse" ]; then
    echo "sparse folder exists in ${data_path}"
else
    mkdir -p "${data_path}/sparse"
    cp -r "${data_path}/hloc_output/sfm/"* "${data_path}/sparse/"
fi

python3 projects/neuralangelo/scripts/convert_data_to_json.py --data_dir ${data_path} --scene_type ${scene_type}
python3 projects/neuralangelo/scripts/generate_config.py --sequence_name ${1} --data_dir ${data_path} --scene_type ${scene_type}
