# bash eval_r3dp.sh block_hammer_beat L515 100 600 0 0 4

DEBUG=False

task_name=${1}
head_camera_type=${2}
expert_data_num=${3}
checkpoint_num=${4}
seed=${5}
gpu_id=${6}
tau=${7}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ../..
python ./script/eval_policy_r3dp.py "$task_name" "$head_camera_type" "$expert_data_num" "$checkpoint_num" "$seed" --tau "$tau"