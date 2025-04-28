export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /home/liyouzhou/tensorflow_datasets/ \
  --dataset_name mikasa_robo_baseline_tfds \
  --run_root_dir ~/study/openvla/logs \
  --batch_size 3 \
  --learning_rate 5e-4 \
  --max_steps 50005 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 8 \
  --save_steps 30000 \
  --wandb_entity "leothemagnificent-university-of-cambridge" \
  --wandb_project "openvla-mikasa-baseline" \
  --run_id_note mikasa_robo_baseline_tfds \
  --grad_accumulation_steps 2
