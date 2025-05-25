#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path ./logs/mikasa_baseline/openvla-7b+mikasa_baseline_mix+b16+lr-5e-05+lora-r16+dropout-0.1--mikasa_baseline_mix--image_aug--25-05-16_19-21-23--10000_chkpt/ \
  --data_root_dir /home/liyouzhou/tensorflow_datasets/ \
  --dataset_name mikasa_baseline_mix \
  --run_root_dir ~/study/openvla/logs \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --max_steps 50005 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 8 \
  --save_steps 10000 \
  --wandb_entity "leothemagnificent-university-of-cambridge" \
  --wandb_project "openvla-mikasa" \
  --run_id_note mikasa_baseline_mix \
  --grad_accumulation_steps 1 \
  --val_frequency 10 \
  --val_test_num_batches 20 \
  --start_step 10000
