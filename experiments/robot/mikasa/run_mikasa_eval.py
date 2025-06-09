"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import torch

import draccus
import numpy as np
import tqdm

import wandb
import gymnasium as gym
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info

from prismatic.models.backbones.llm.prompting.base_prompter import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import RLDSBatchTransform

from transformers.modeling_outputs import CausalLMOutputWithPast

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import crop_and_resize

import tensorflow as tf
from PIL import Image
import imageio


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")

    for i, img in enumerate(rollout_images):
        imageio.imwrite(
            f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}--frame={i}.png",
            img,
        )

    return mp4_path


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "mikasa"                  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 0                          # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    num_envs: int = 4                                # Number of environments to run in parallel (for multi-agent tasks)
    #################################################################################################################
    # fmt: on


TEST_SUITES = {
    "mikasa": {
        "tasks": [
            {
                "task_name": "RememberColor3-v0",
                "env_name": "RememberColor3-v0",
                "baseline_prompt": False,
            },
            {
                "task_name": "RememberColor3-v0_baseline",
                "env_name": "RememberColor3-v0",
                "baseline_prompt": True,
            },
        ],
    }
}


def center_crop(image, batch_size=1, crop_scale=0.9, return_pil_image=False):
    image = Image.fromarray(image)
    image = image.convert("RGB")

    # Convert to TF Tensor and record original data type (should be tf.uint8)
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    # Convert to data type tf.float32 and values between [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop and then resize back to original size
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert back to PIL Image
    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")

    if return_pil_image:
        return image

    image = np.array(image)

    return image


def predict_action(
    model, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
) -> np.ndarray:
    """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
    # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    if not torch.all(input_ids[:, -1] == 29871):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        input_ids = torch.cat(
            (input_ids, torch.Tensor([29871]).long().repeat(batch_size, 1).to(input_ids.device)), dim=1
        )

    # Run VLA inference
    generated_ids = model.generate(input_ids, max_new_tokens=model.get_action_dim(unnorm_key), **kwargs)

    batch_actions = []
    for i in range(generated_ids.shape[0]):
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[i, -model.get_action_dim(unnorm_key) :].cpu().numpy()
        discretized_actions = model.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
        normalized_actions = model.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = model.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        batch_actions.append(actions)

    # Stack actions and return
    actions = np.stack(batch_actions, axis=0)

    return actions


def infer_batch(images, prompts, model, processor, unnorm_key, crop_scale=0.9):
    """Infer a batch of samples."""
    batch_size = len(images)
    assert len(prompts) == batch_size, "Number of prompts must match number of images!"

    prompts = [f"In: What action should the robot take to {prompt.lower()}?\nOut:" for prompt in prompts]

    # Center crop images if necessary
    if crop_scale < 1 and crop_scale > 0:
        images = [center_crop(image, crop_scale=crop_scale, return_pil_image=True) for image in images]

    # Process inputs.
    input = processor(prompts, images).to("cuda", dtype=torch.bfloat16)

    # Get action.
    actions = [
        model.predict_action(
            input_ids=input["input_ids"][i].unsqueeze(0),
            attention_mask=input["attention_mask"][i].unsqueeze(0),
            pixel_values=input["pixel_values"][i].unsqueeze(0),
            unnorm_key=unnorm_key,
            do_sample=False,
        )
        for i in range(batch_size)
    ]

    actions = np.array(actions)

    return actions


@draccus.wrap()
def eval_mikasa(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
        )
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    num_tasks_in_suite = len(TEST_SUITES[cfg.task_suite_name]["tasks"])

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        env_name = TEST_SUITES[cfg.task_suite_name]["tasks"][task_id]["env_name"]
        task_name = TEST_SUITES[cfg.task_suite_name]["tasks"][task_id]["task_name"]
        print(f"Running task {task_name}...")
        num_envs = cfg.num_envs
        env_kwargs_rgb = dict(
            num_envs=num_envs,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pose",
            render_mode="all",
            sim_backend="gpu",
            reward_mode="normalized_dense",
        )
        unnorm_key = f"mikasa_robo_tfds/{env_name}"  # Action un-normalization key for OpenVLA
        # [OpenVLA] Check that the model contains the action un-normalization key
        assert (
            unnorm_key in model.norm_stats
        ), f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`! valid keys: {model.norm_stats.keys()}"

        env = gym.make(env_name, **env_kwargs_rgb)
        state_wrappers_list, episode_timeout = env_info(env_name)
        print(f"Episode timeout: {episode_timeout}")
        for wrapper_class, wrapper_kwargs in state_wrappers_list:
            env = wrapper_class(env, **wrapper_kwargs)

        # Start episodes
        task_episodes, task_successes = 0, 0
        dist_to_target = []
        all_rewards = []
        for episode_idx in tqdm.trange(0, cfg.num_trials_per_task, num_envs):
            # Reset environment
            obs, info = env.reset()

            oracle_info = [int(x) for x in info["oracle_info"].cpu()]
            colors = [["red", "green", "blue"][x] for x in oracle_info]
            prompts = [
                (
                    f"Touch the {color} cube"
                    if TEST_SUITES[cfg.task_suite_name]["tasks"][task_id]["baseline_prompt"]
                    else "Memorize the the colors of the cube shown on the table, and then touch the same coloured cube out of all the cubes."
                )
                for color in colors
            ]

            log_file.write(f"\nTask: {prompts}\n")

            # Setup
            t = 0
            replay_images = [[] for _ in range(num_envs)]
            max_steps = 500  # default max steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            terminated_flags = np.array([False] * num_envs)
            truncated_flags = np.array([False] * num_envs)
            success_flags = np.array([False] * num_envs)
            final_rewards = np.zeros(num_envs)
            final_distances = np.zeros(num_envs)

            while t < max_steps + cfg.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    action = env.action_space.sample()
                    action = np.zeros(action.shape)
                    obs, reward, terminated, truncated, info = env.step(action)
                    t += 1
                    continue

                # Get observation image
                images = obs["sensor_data"]["base_camera"]["rgb"]
                images = images.cpu().numpy()

                # Save preprocessed image for replay video
                for i in range(num_envs):
                    replay_images[i].append(images[i])

                # query VLA model for action
                actions = infer_batch(
                    images=images,
                    prompts=prompts,
                    model=model,
                    processor=processor,
                    unnorm_key=unnorm_key,
                    crop_scale=0.9 if cfg.center_crop else 1.0,
                )
                actions = torch.from_numpy(actions)
                actions = actions * 10
                obs, reward, terminated, truncated, info = env.step(actions)

                for i in range(num_envs):
                    if terminated[i].cpu().numpy():
                        terminated_flags[i] = True
                        final_rewards[i] = reward[i].cpu().numpy()
                        final_distances[i] = info["reward_dict"]["tcp_to_obj_dist"][i].cpu().numpy()
                    if truncated[i].cpu().numpy():
                        truncated_flags[i] = True
                        final_rewards[i] = reward[i].cpu().numpy()
                        final_distances[i] = info["reward_dict"]["tcp_to_obj_dist"][i].cpu().numpy()
                    if info["success"][i].cpu().numpy():
                        success_flags[i] = True

                if all(terminated_flags | truncated_flags):
                    # print("All environments terminated or truncated, ending episode.")
                    break
                t += 1

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_successes += np.sum(success_flags)
            total_successes += np.sum(success_flags)

            # Save a replay video of the episode
            for i in range(num_envs):
                task_episodes += 1
                total_episodes += 1

                mp4_path = save_rollout_video(
                    replay_images[i],
                    total_episodes,
                    success=terminated_flags[i],
                    task_description=task_name,
                    log_file=log_file,
                )

                if cfg.use_wandb:
                    wandb.log(
                        {
                            f"rollout_video/{task_name}/{colors[i]}": wandb.Video(mp4_path, format="mp4"),
                            f"sucess": success_flags[i],
                            f"distance_to_target": final_distances[i],
                            f"reward": final_rewards[i],
                            f"episode_idx": task_episodes - 1,
                        }
                    )

                dist_to_target.append(final_distances[i])
                all_rewards.append(final_rewards[i])

            # Log current results
            print(f"Success: {terminated}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {terminated}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        avg_dist_to_target = np.mean(dist_to_target)
        average_reward = np.mean(all_rewards)

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        print(f"Average Distance to target: {avg_dist_to_target}")
        print(f"Average Reward: {average_reward}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.write(f"Average Distance to target: {avg_dist_to_target}\n")
        log_file.write(f"Average Reward: {average_reward}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"{env_name}/success_rate": float(task_successes) / float(task_episodes),
                    f"{env_name}/num_episodes": task_episodes,
                    f"{env_name}/average_distance": avg_dist_to_target,
                    f"{env_name}/average_reward": average_reward,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_mikasa()
